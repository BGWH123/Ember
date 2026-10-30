English | [中文](./README_CN.md)

# Ember

Build the internals of modern AI systems from scratch. Ember is a local-first desktop practice environment with 181 PyTorch exercises, learning paths, theory questions, and mock interviews. It uses Tauri 2, React, and a Rust core.

Browsing exercises and reference solutions needs no Python runtime. Install a separate local evaluator module to run code.

## Desktop development

Install Node.js 22+, Rust stable, and the [Tauri platform prerequisites](docs/fern/pages/desktop/installation/install-desktop.mdx). Windows builds use the MSVC toolchain and Visual Studio C++ Build Tools.

    npm ci
    npm run dev

Noto Sans, Noto Sans SC, and Fira Code are bundled for offline use. Font licenses ship in apps/desktop/public/licenses.

## Source evaluator

Windows PowerShell:

    uv venv --python 3.12 .venv
    uv pip install --python .venv/Scripts/python.exe -r modules/evaluator/requirements.txt
    uv pip install --python .venv/Scripts/python.exe pytest
    $env:EMBER_DEV_PYTHON = (Resolve-Path .venv/Scripts/python.exe).Path
    npm run dev

On Linux/macOS use .venv/bin/python and set the same environment variable using your shell syntax. The source interpreter is accepted only in debug builds; release applications use separately signed evaluator modules. See the [development guide](docs/fern/pages/developer-guide/development/build-from-source.mdx).

## Validation and packaging

    npm run check:core
    npm run test:python
    npm run build:ui
    cargo build --release -p ember-core --features developer-tools
    npm --prefix apps/desktop exec -- playwright install chromium
    npm --prefix apps/desktop test
    npm run build:windows

Windows installers appear under target/release/bundle. Use npm run build:macos on macOS. See the [packaging guide](docs/fern/pages/developer-guide/development/package-installers.mdx).

## Architecture

| Directory | Responsibility |
| --- | --- |
| apps/desktop | React UI and Tauri shell |
| crates/ember-core | Catalog, SQLite records, and AI requests |
| crates/ember-evaluator | Local evaluator protocol, module installation and signatures |
| modules/evaluator | Standalone Python evaluator |
| content/catalog | Bundled catalog and reference solutions |
| torch_judge/tasks | Executable task definitions |
| scripts/desktop | Export, build, validation, and performance tools |

The legacy Next.js frontend and duplicate grading_service backend have been removed. Desktop requests use Tauri IPC and the Rust core. Evaluation runs through a bounded local subprocess protocol without an HTTP service.

Online evaluation is unavailable. Only an inactive HTTP boundary remains in crates/ember-evaluator/src/remote.rs, marked TODO for future online evaluation. It never sends requests or submitted code. The Python service and its web-server dependencies have been removed.

Refresh catalog data with npm run build:problems and npm run build:solutions. Authored presentation metadata stays in content/catalog. Read-only import of old learning records remains available in Settings.

## Documentation and credits

[Desktop documentation](docs/README.md) covers installation, modules, architecture, and performance tools. Preview with npm run docs:install and npm run docs:dev; validate with npm run docs:check.

Based on [TorchCode](https://github.com/duoan/TorchCode). Original attribution is preserved in [LICENSE](LICENSE). MIT License.
