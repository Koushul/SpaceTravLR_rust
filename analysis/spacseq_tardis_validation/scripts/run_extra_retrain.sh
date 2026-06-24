#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GENES=$(ls runs/baseline_pooled_seed/*_betadata.feather | sed 's|.*/||;s|_betadata.feather||' | paste -sd,)
echo "Training ${GENES//,/, } genes count: $(echo "$GENES" | tr ',' '\n' | wc -l)"
export SPACETRAVLR_FORCE_KEEP_GENES="$GENES"
export SPACETRAVLR_FORCE_CPU=1
spacetravlr --plain --training-mode seed \
  --config spaceship_config_pooled_extra.toml \
  --h5ad data/pooled/baseline_ntc.h5ad \
  --output-dir runs/baseline_pooled_extra_seed \
  --max-ligands 200 \
  --genes "$GENES" \
  --parallel 8
