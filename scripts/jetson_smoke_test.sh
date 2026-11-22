#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/128x128.png" >&2
    exit 2
fi
cd "$ROOT"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/tinysr-jetson --input "$1" --output outputs/smoke.png --warmup 1 --runs 3
file outputs/smoke.png
