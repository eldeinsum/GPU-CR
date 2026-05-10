#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
smoke_bin="${TMPDIR:-/tmp}/gpucr_ckpt_smoke"
smoke_log="${TMPDIR:-/tmp}/gpucr_ckpt_smoke.log"

cargo build --manifest-path "$repo_root/rust/Cargo.toml" --workspace
nvcc -o "$smoke_bin" "$repo_root/rust/smoke/cuda_ckpt_smoke.cu"

mkdir -p /mnt/huge-ckpt
rm -f "$smoke_log" /tmp/gpucr_restore_go_* /mnt/huge-ckpt/rust-smoke-*

LD_PRELOAD="$repo_root/rust/target/debug/libgpucr_preload.so" "$smoke_bin" >"$smoke_log" 2>&1 &
app=$!
cleanup() {
    touch "/tmp/gpucr_restore_go_$app" 2>/dev/null || true
    kill "$app" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 100); do
    if rg -q '^READY ' "$smoke_log"; then
        break
    fi
    sleep 0.1
done
cat "$smoke_log"
if ! rg -q '^READY ' "$smoke_log"; then
    echo "smoke app did not become ready" >&2
    exit 11
fi

timeout 45s "$repo_root/rust/target/debug/gpucr-client" checkpoint "$app" "/mnt/huge-ckpt/rust-smoke-$app"
timeout 45s "$repo_root/rust/target/debug/gpucr-client" restore "$app"
touch "/tmp/gpucr_restore_go_$app"
wait "$app"
trap - EXIT
cat "$smoke_log"
