#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$project_root"
docker build -t ember-desktop-build -f scripts/desktop/Dockerfile .
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$project_root:/workspace" -v "${HOME}/.rustup:${HOME}/.rustup:ro" \
  -v "${HOME}/.cargo:${HOME}/.cargo" \
  -e HOME="$HOME" -e RUSTUP_HOME="${HOME}/.rustup" -e CARGO_HOME="${HOME}/.cargo" \
  -e PATH="${HOME}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  ember-desktop-build cargo build -p ember-desktop --locked --features custom-protocol --target-dir artifacts/desktop/native-target
