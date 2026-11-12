#!/usr/bin/env bash
set -euo pipefail
ROOT="${LAB_ROOT:-/workspace}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/repos"
ALCHEMY_REV=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["source"]["alchemy_git"])' "$HERE/config/revisions.json")
TRELLIS_REV=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["source"]["trellis_git"])' "$HERE/config/revisions.json")
for spec in "Alchemy3D|https://github.com/libd1/Alchemy3D.git|${ALCHEMY_REV}" "TRELLIS.2|https://github.com/microsoft/TRELLIS.2.git|${TRELLIS_REV}"; do
  IFS='|' read -r name url rev <<< "$spec"
  if [[ ! -d "$ROOT/repos/$name/.git" ]]; then git clone --filter=blob:none "$url" "$ROOT/repos/$name"; fi
  git -C "$ROOT/repos/$name" fetch origin "$rev" --depth 1
  git -C "$ROOT/repos/$name" checkout --detach "$rev"
  if [[ "$name" == "TRELLIS.2" ]]; then git -C "$ROOT/repos/$name" submodule update --init --recursive || true; fi
  echo "[PIN] $name: $(git -C "$ROOT/repos/$name" rev-parse HEAD)"
done
