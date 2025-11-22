#!/usr/bin/env bash
set -euo pipefail

: "${USER_UID:=1000}"
: "${USER_GID:=1000}"
: "${USERNAME:=cli}"

# Activate venv for login shells and non-interactive shells
export VIRTUAL_ENV=/opt/venv
export PATH="$VIRTUAL_ENV/bin:$PATH"

# If user mounts code to /workspace, ensure ownership doesn’t break builds

# Create group if missing
if ! getent group "${USER_GID}" >/dev/null 2>&1; then
  groupadd -g "${USER_GID}" "${USERNAME}" || groupadd -g "${USER_GID}" "grp${USER_GID}" || true
fi

# Create user if missing
if ! id -u "${USER_UID}" >/dev/null 2>&1; then
  useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USERNAME}" || true
fi

# Ensure home exists
HOME_DIR="$(getent passwd "${USER_UID}" | cut -d: -f6)"
mkdir -p "${HOME_DIR}"

# Make sure our common writable paths are owned (skip bind mounts like /workspace)
for d in /opt/venv; do
  if [ -d "$d" ]; then chown -R "${USER_UID}:${USER_GID}" "$d" || true; fi
done

# Export editor defaults for the user
echo 'export EDITOR=nvim; export VISUAL=nvim' >> "${HOME_DIR}/.bashrc" || true
chown "${USER_UID}:${USER_GID}" "${HOME_DIR}/.bashrc" || true

# Drop privileges   (use tini/gosu if installed; otherwise su-exec/ runuser)
if command -v gosu >/dev/null 2>&1; then
  exec gosu "${USER_UID}:${USER_GID}" "$@"
else
  exec runuser -u "$(id -nu "${USER_UID}")" -- "$@"
fi

# Print helpful banner
echo "Dev container ready. Python: $(python --version)"
echo "CUDA version: $(nvcc --version | sed -n 's/^.*release \(.*\),.*/\1/p')"
exec "$@"
