#!/usr/bin/env bash
set -euo pipefail

# The public uv.lock is authoritative. Export its NVIDIA training branch, then
# install the same pinned artifacts using Alibaba Cloud's private mirrors.
# This does not re-resolve the other branches or rewrite uv.lock.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

pypi_index="${PAI_PYPI_INDEX:-http://mirrors.cloud.aliyuncs.com/pypi/simple/}"
cu126_links="${PAI_CUDA_WHEELS:-http://mirrors.cloud.aliyuncs.com/pytorch-wheels/cu126/}"
requirements="$(mktemp --suffix=.txt)"
trap 'rm -f "$requirements"' EXIT

uv export --locked --only-group nvidia-train --no-emit-project --no-emit-index-url --no-annotate --no-header --output-file "$requirements" >/dev/null
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.12 .venv
fi

uv pip sync --python "$root/.venv/bin/python" --default-index "$pypi_index" --find-links "$cu126_links" --require-hashes "$requirements" "$@"
