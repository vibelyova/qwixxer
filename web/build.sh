#!/bin/bash
set -e
cd "$(dirname "$0")/.."
wasm-pack build web --target web
wasm-opt web/pkg/qwixxer_web_bg.wasm -o web/pkg/qwixxer_web_bg.wasm -Oz \
  --enable-bulk-memory --enable-simd --enable-mutable-globals \
  --enable-sign-ext --enable-nontrapping-float-to-int
echo "WASM built: $(du -h web/pkg/qwixxer_web_bg.wasm | cut -f1)"
