# Qwixx Web UI

Browser-based Qwixx game and state explorer, powered by WASM.

## Prerequisites

- Rust toolchain with `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/): `cargo install wasm-pack`
- [binaryen](https://github.com/WebAssembly/binaryen) (for wasm-opt): `apt install binaryen`
- Node.js 18+

## Setup

```bash
cd web
npm install
```

## Build WASM

The build script compiles the Rust WASM crate and runs wasm-opt for size optimization:

```bash
./web/build.sh
```

This embeds `champion.txt` (GA weights) and `dqn_model/model.mpk` (DQN weights) into the WASM binary via `include_bytes!`. Rebuild after changing either file.

## Development server

```bash
cd web
npx vite --port 3000
```

- Game: http://localhost:3000/qwixxer/
- State Explorer: http://localhost:3000/qwixxer/explorer.html

## Production build

```bash
cd web
npx vite build
```

Output in `web/dist/`, ready for static hosting (GitHub Pages).

## Deploy to GitHub Pages

Push to `main` — the GitHub Actions workflow (`.github/workflows/deploy.yml`) builds and deploys automatically.
