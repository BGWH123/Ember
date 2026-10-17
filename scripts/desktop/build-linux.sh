#!/usr/bin/env bash
# Build an unsigned Linux .deb without installing native dependencies on the host.
set -euo pipefail
project_root="$(cd "$(dirname "$0")/../.." && pwd)"
node_prefix="$(dirname "$(dirname "$(command -v node)")")"
cd "$project_root"
docker build -t ember-desktop-build -f scripts/desktop/Dockerfile .
module_key_args=()
if [[ -n "${EMBER_EVALUATOR_PUBLIC_KEY:-}" ]]; then
  module_key_args=(-e "EMBER_EVALUATOR_PUBLIC_KEY=$EMBER_EVALUATOR_PUBLIC_KEY")
fi
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$project_root:/workspace" -v "$node_prefix:/opt/node:ro" \
  -v "${HOME}/.rustup:${HOME}/.rustup:ro" -v "${HOME}/.cargo:${HOME}/.cargo" \
  -e HOME="$HOME" -e RUSTUP_HOME="${HOME}/.rustup" -e CARGO_HOME="${HOME}/.cargo" \
  -e CARGO_TARGET_DIR=/workspace/artifacts/desktop/native-target \
  -e PATH="/opt/node/bin:${HOME}/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  "${module_key_args[@]}" ember-desktop-build \
  npm --prefix apps/desktop run tauri -- build --bundles deb \
    --config '{"build":{"beforeBuildCommand":""}}'
