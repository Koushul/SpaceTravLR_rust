#!/usr/bin/env bash
set -euo pipefail

REPO_ID="Koushul/spacetravlr"
REPO_TYPE="dataset"
SOURCE_ROOT="/ix/djishnu/shared/djishnu_kor11/rust_trainings/xenium_skin_mixed/conditions"
HUB_PREFIX="xenium_skin_mixed"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGING_DIR="${SCRIPT_DIR}/../data/hf_spacetravlr"
LOG_DIR="${STAGING_DIR}/upload_logs"
SAMPLES=(sample12 sample13 sample14 sample15 sample16)
EXCLUDE_GLOB="**/perturbations/**"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*"
}

require_hf_auth() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  fi

  if hf auth whoami >/dev/null 2>&1; then
    log "Authenticated as $(hf auth whoami 2>/dev/null | tr '\n' ' ')"
    return 0
  fi

  if [[ -f "${HOME}/.git-credentials" ]]; then
    local token
    token="$(python3 - <<'PY'
import re
from pathlib import Path
for line in Path.home().joinpath(".git-credentials").read_text().splitlines():
    if "huggingface.co" not in line:
        continue
    m = re.match(r"https://[^:]+:([^@]+)@huggingface\.co", line)
    if m:
        print(m.group(1))
        break
PY
)"
    if [[ -n "${token}" ]]; then
      export HF_TOKEN="${token}"
      export HUGGING_FACE_HUB_TOKEN="${token}"
      if hf auth whoami >/dev/null 2>&1; then
        log "Authenticated via ~/.git-credentials"
        return 0
      fi
    fi
  fi

  cat >&2 <<'EOF'
ERROR: Hugging Face write access is required.

Set a write token, then rerun:
  export HF_TOKEN=hf_...
  bash scripts/upload_xenium_skin_to_hf.sh [all|metadata|sample12|...]

Create a token at https://huggingface.co/settings/tokens with write access to Koushul/spacetravlr.
EOF
  exit 1
}

upload_path() {
  local local_path="$1"
  local remote_path="$2"
  local message="$3"
  local log_file="${LOG_DIR}/$(echo "${remote_path}" | tr '/' '_').log"

  log "Uploading ${local_path} -> ${REPO_ID}:${remote_path}"
  hf upload "${REPO_ID}" "${local_path}" "${remote_path}" \
    --repo-type "${REPO_TYPE}" \
    --exclude "${EXCLUDE_GLOB}" \
    --commit-message "${message}" \
    2>&1 | tee "${log_file}"
}

upload_metadata() {
  upload_path "${STAGING_DIR}/README.md" "README.md" "Add dataset README for xenium_skin_mixed cohort"
  upload_path "${STAGING_DIR}/manifest.json" "${HUB_PREFIX}/manifest.json" "Add xenium_skin_mixed manifest"
  upload_path "${SOURCE_ROOT}/run.toml" "${HUB_PREFIX}/run.toml" "Add shared xenium_skin_mixed run config"

  printf '%s\n' '*.feather filter=lfs diff=lfs merge=lfs -text' > "${STAGING_DIR}/gitattributes.feather-snippet"
  upload_path "${STAGING_DIR}/gitattributes.feather-snippet" ".gitattributes.feather-snippet" "Document recommended LFS tracking for feather outputs"
}

upload_sample() {
  local sample="$1"
  local local_dir="${SOURCE_ROOT}/${sample}"
  local remote_dir="${HUB_PREFIX}/${sample}"

  if [[ ! -d "${local_dir}" ]]; then
    log "Missing sample directory: ${local_dir}"
    exit 1
  fi

  upload_path "${local_dir}" "${remote_dir}" "Add xenium_skin_mixed ${sample} SpaceTravLR outputs (no perturbations)"
}

main() {
  local target="${1:-all}"

  command -v hf >/dev/null 2>&1 || {
    echo "ERROR: hf CLI not found. Install with: python3 -m pip install huggingface_hub" >&2
    exit 1
  }

  require_hf_auth

  case "${target}" in
    metadata)
      upload_metadata
      ;;
    all)
      upload_metadata
      for sample in "${SAMPLES[@]}"; do
        upload_sample "${sample}"
      done
      ;;
    sample12|sample13|sample14|sample15|sample16)
      upload_sample "${target}"
      ;;
    *)
      echo "Usage: $0 [all|metadata|sample12|sample13|sample14|sample15|sample16]" >&2
      exit 1
      ;;
  esac

  log "Done."
}

main "$@"
