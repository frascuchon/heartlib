"""Rotate the heartmula-api-key Modal Secret and redeploy the service."""

import secrets
import subprocess

MODAL_CLI = ".venv/bin/modal"
SECRET_NAME = "heartmula-api-key"
ENV_VAR_NAME = "HEARTMULA_API_KEY"
SERVICE_SCRIPT = "scripts/serve_modal.py"

print("Generando nueva API key...")
new_key = secrets.token_urlsafe(32)
print(f"Nueva API key: {new_key}")
print()
print("⚠  Guarda esta clave ahora — no se puede recuperar después.")
print()

print(f"Actualizando Modal Secret '{SECRET_NAME}'...")
subprocess.run(
    [MODAL_CLI, "secret", "create", SECRET_NAME, f"{ENV_VAR_NAME}={new_key}", "--force"],
    check=True,
)
print("✓ Secret actualizado.")
print()

print("Redesplegando servicio...")
subprocess.run([MODAL_CLI, "deploy", SERVICE_SCRIPT], check=True)
print()
print("✓ Servicio redesplegado con la nueva API key.")
