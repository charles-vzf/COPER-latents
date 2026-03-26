#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-coper"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"

RECREATE=1
if [[ "${1:-}" == "--keep" ]]; then
  RECREATE=0
fi

if [[ -d "${VENV_DIR}" && "${RECREATE}" -eq 1 ]]; then
  rm -rf "${VENV_DIR}"
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

if ! [[ -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Missing requirements.txt: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

# Install torch separately because it needs an index URL depending on CUDA/CPU.
# We pin to the same major version as the current environment when possible.
TORCH_VERSION="2.10.0"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Detected nvidia-smi: installing CUDA torch (${TORCH_VERSION}+cu128)"
  "${VENV_DIR}/bin/pip" install \
    --index-url "https://download.pytorch.org/whl/cu128" \
    "torch==${TORCH_VERSION}+cu128"
else
  echo "No nvidia-smi detected: installing CPU torch (${TORCH_VERSION})"
  "${VENV_DIR}/bin/pip" install "torch==${TORCH_VERSION}"
fi

"${VENV_DIR}/bin/pip" install -r "${REQUIREMENTS_FILE}"

echo "Done. Activate with:"
echo "  source \"${VENV_DIR}/bin/activate\""

