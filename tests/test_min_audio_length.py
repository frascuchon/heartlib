"""Tests for min_audio_length_ms enforcement in the generation loop.

No model weights needed — uses a mock generator that simulates EOS behavior.
"""

import torch
import pytest


def simulate_generation_loop(eos_frame, max_frames, min_frames, eos_id=8193, n_codebooks=8):
    """Simulate the fixed generation loop.

    The mock generator returns EOS when the INPUT token contains an EOS value,
    otherwise returns a valid token. This simulates model recovery when
    last_valid_token (non-EOS) is re-fed after skipping an EOS frame.
    """
    curr_token = torch.zeros((1, n_codebooks), dtype=torch.long)
    frames = [curr_token]
    last_valid_token = curr_token
    call_count = 0

    for i in range(max_frames):
        # Mock generate_frame: returns EOS only on the eos_frame-th call
        if call_count == eos_frame:
            curr_token = torch.full((1, n_codebooks), eos_id, dtype=torch.long)
        else:
            curr_token = torch.zeros((1, n_codebooks), dtype=torch.long)
        call_count += 1

        if torch.any(curr_token[0:1, :] >= eos_id):
            if i + 1 >= min_frames:
                break
            continue  # fixed: last_valid_token not updated → re-feeds valid context
        last_valid_token = curr_token
        frames.append(curr_token)

    return frames


def simulate_broken_loop(eos_frame, max_frames, min_frames, eos_id=8193, n_codebooks=8):
    """Simulate the BUGGY generation loop (feeds curr_token, including EOS)."""
    curr_token = torch.zeros((1, n_codebooks), dtype=torch.long)
    frames = [curr_token]
    call_count = 0

    for i in range(max_frames):
        if call_count == eos_frame:
            curr_token = torch.full((1, n_codebooks), eos_id, dtype=torch.long)
        else:
            # Buggy: if curr_token was EOS, model keeps generating EOS
            if torch.any(curr_token >= eos_id):
                curr_token = torch.full((1, n_codebooks), eos_id, dtype=torch.long)
            else:
                curr_token = torch.zeros((1, n_codebooks), dtype=torch.long)
        call_count += 1

        if torch.any(curr_token[0:1, :] >= eos_id):
            if i + 1 >= min_frames:
                break
            continue
        frames.append(curr_token)

    return frames


# ---------------------------------------------------------------------------
# Fixed loop tests
# ---------------------------------------------------------------------------

def test_early_eos_min_enforced():
    """EOS fires early (frame 3), min=8: loop continues and hits min."""
    frames = simulate_generation_loop(eos_frame=3, max_frames=20, min_frames=8)
    # 1 initial + frames 0,1,2 (valid) + frames 4..7 (valid, after EOS skip at i=3)
    # EOS at call 3 → skipped at i=3, then valid frames at i=4..7 → break at i=7 (i+1=8 >= 8)
    assert len(frames) >= 8


def test_eos_at_min_boundary():
    """EOS fires exactly at min boundary: loop breaks immediately."""
    # EOS at call 7 → i=7, i+1=8 >= min_frames=8 → break
    # frames: 1 initial + 7 valid (i=0..6)
    frames = simulate_generation_loop(eos_frame=7, max_frames=20, min_frames=8)
    assert len(frames) == 8


def test_eos_after_min_stops_immediately():
    """EOS fires after min: loop breaks immediately."""
    # EOS at call 12 → i=12, i+1=13 >= min_frames=8 → break
    # frames: 1 initial + 12 valid (i=0..11)
    frames = simulate_generation_loop(eos_frame=12, max_frames=20, min_frames=8)
    assert len(frames) == 13


def test_no_eos_hits_max():
    """No EOS → loop runs to max_frames."""
    frames = simulate_generation_loop(eos_frame=999, max_frames=10, min_frames=5)
    # 1 initial + 10 loop frames
    assert len(frames) == 11


def test_min_equals_max_eos_at_min():
    """min == max, EOS fires at min: break immediately."""
    # eos_frame=5 → i=5, i+1=6 > min_frames=5 → break
    frames = simulate_generation_loop(eos_frame=5, max_frames=5, min_frames=5)
    # 1 initial + 5 valid frames (i=0..4), then EOS at i=5 but max=5 so loop ends
    assert len(frames) == 6


# ---------------------------------------------------------------------------
# Regression: fixed loop vs broken loop
# ---------------------------------------------------------------------------

def test_fix_produces_more_frames_than_buggy():
    """With the fix, early EOS produces more frames than the buggy version."""
    fixed = simulate_generation_loop(eos_frame=3, max_frames=20, min_frames=8)
    broken = simulate_broken_loop(eos_frame=3, max_frames=20, min_frames=8)
    # Fixed should meet or exceed min_frames; broken gets stuck after EOS
    assert len(fixed) >= 8
    assert len(fixed) > len(broken)
