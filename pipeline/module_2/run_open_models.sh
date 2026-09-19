#!/usr/bin/env bash
# pipeline/module_2/run_open_models.sh - Run the open-weight models on a GPU pod.
#
# One command per pod session, from a fresh clone of this repository:
#
#     git clone https://github.com/hugorydel/VLM_Illusion_Analysis.git
#     cd VLM_Illusion_Analysis
#     git checkout <commit>        # optional: pin the exact code to run
#     bash pipeline/module_2/run_open_models.sh --pilot qwen3-vl-2b internvl3.5-2b
#     bash pipeline/module_2/run_open_models.sh qwen3-vl-2b qwen3-vl-8b
#
# In order, it:
#   1. refuses to run on modified tracked files (--allow-dirty overrides), so
#      every result can be traced to a commit;
#   2. unpacks stimuli.tar.gz (repo root or /workspace) if stimuli/ is absent;
#   3. installs the Python dependencies (VLLM_SPEC; --skip-install to reuse);
#   4. checks the pod has the GPUs each model needs (config.MODELS n_gpus);
#   5. runs pipeline/module_2/local_vlm.py for each model, logging its output
#      to run.log beside the results;
#   6. records the environment (commit, pip freeze, nvidia-smi) in
#      environment.txt beside the results;
#   7. packs every model's outputs into one archive to copy back.
#
# Outputs go to results/<model>/, or results/_pilot/<model>/ with --pilot.
# A model that fails is reported at the end; the others still run.

set -euo pipefail

# vLLM >= 0.11 is the first release with Qwen3-VL. Pin this to the exact
# version the pilot ran on before the full runs; run_info.json and
# environment.txt record the installed version either way.
VLLM_SPEC="${VLLM_SPEC:-vllm>=0.11}"

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

pilot=0
allow_dirty=0
skip_install=0
models=()
for arg in "$@"; do
    case "$arg" in
        --pilot) pilot=1 ;;
        --allow-dirty) allow_dirty=1 ;;
        --skip-install) skip_install=1 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $arg" >&2; exit 2 ;;
        *) models+=("$arg") ;;
    esac
done
if [ "${#models[@]}" -eq 0 ]; then
    echo "Name at least one model, e.g.: bash $0 --pilot qwen3-vl-2b" >&2
    exit 2
fi

log() { printf '\n== %s\n' "$*"; }

# 1. Code state ---------------------------------------------------------------
commit="$(git rev-parse HEAD)"
dirty=no
if ! git diff --quiet || ! git diff --cached --quiet; then
    dirty=yes
    if [ "$allow_dirty" -eq 0 ]; then
        echo "Tracked files are modified, so results could not be traced to commit $commit." >&2
        echo "Commit or discard the changes, or pass --allow-dirty." >&2
        exit 1
    fi
    echo "WARNING: running on modified files; results are not reproducible from $commit alone."
fi
log "Code: commit $commit"

# 2. Stimuli ------------------------------------------------------------------
if [ ! -d stimuli ]; then
    for archive in "$REPO_ROOT/stimuli.tar.gz" /workspace/stimuli.tar.gz; do
        if [ -f "$archive" ]; then
            log "Unpacking $archive"
            tar -xzf "$archive" -C "$REPO_ROOT"
            break
        fi
    done
fi
if [ ! -d stimuli ]; then
    echo "No stimuli/ folder and no stimuli.tar.gz in $REPO_ROOT or /workspace." >&2
    exit 1
fi

# Model weights go on the pod's volume when it has one, not the container disk.
if [ -d /workspace ]; then
    export HF_HOME="${HF_HOME:-/workspace/hf}"
fi
export PYTHONUNBUFFERED=1

# 3. Dependencies -------------------------------------------------------------
if [ "$skip_install" -eq 0 ]; then
    log "Installing $VLLM_SPEC, pillow, pandas"
    python -m pip install --quiet "$VLLM_SPEC" pillow pandas
fi
python -c "import vllm; print('vLLM', vllm.__version__)"

# 4. GPUs ---------------------------------------------------------------------
available_gpus="$(nvidia-smi -L | wc -l)"
for m in "${models[@]}"; do
    need="$(python -c "
import sys
from config import MODELS
hits = [x for x in MODELS if x['key'] == sys.argv[1] and x.get('backend') == 'vllm']
print(hits[0].get('n_gpus', 1) if hits else 'unknown')
" "$m")"
    if [ "$need" = "unknown" ]; then
        echo "Unknown open model: $m (see config.MODELS)" >&2
        exit 2
    fi
    if [ "$need" -gt "$available_gpus" ]; then
        echo "$m needs $need GPUs; this pod has $available_gpus." >&2
        exit 1
    fi
done
log "GPUs: $available_gpus available"

# 5-6. Runs -------------------------------------------------------------------
if [ "$pilot" -eq 1 ]; then out_base="results/_pilot"; flag="--pilot"; else out_base="results"; flag=""; fi

failed=()
outputs=()
for m in "${models[@]}"; do
    out="$out_base/$m"
    mkdir -p "$out"
    {
        echo "commit: $commit"
        echo "modified tracked files: $dirty"
        echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "runpod_pod_id: ${RUNPOD_POD_ID:-n/a}"
        echo "vllm_spec: $VLLM_SPEC"
        echo
        echo "--- nvidia-smi"
        nvidia-smi
        echo
        echo "--- pip freeze"
        python -m pip freeze
    } > "$out/environment.txt"

    log "Running $m${flag:+ (pilot)}"
    if python -m pipeline.module_2.local_vlm --model "$m" $flag 2>&1 | tee "$out/run.log"; then
        outputs+=("$out")
    else
        failed+=("$m")
        echo "!! $m failed; see $out/run.log"
    fi
done

# 7. Archive ------------------------------------------------------------------
if [ "${#outputs[@]}" -gt 0 ]; then
    dest="/workspace"
    [ -d "$dest" ] || dest="$REPO_ROOT"
    archive="$dest/open_results$([ "$pilot" -eq 1 ] && echo _pilot)_$(date -u +%Y%m%d_%H%M%S).tar.gz"
    tar -czf "$archive" "${outputs[@]}"
    log "Packed ${outputs[*]} into $archive"
    if [ -n "${RUNPOD_PUBLIC_IP:-}" ] && [ -n "${RUNPOD_TCP_PORT_22:-}" ]; then
        echo "Copy it back from your machine with:"
        echo "  scp -P $RUNPOD_TCP_PORT_22 root@$RUNPOD_PUBLIC_IP:$archive ."
    fi
fi

if [ "${#failed[@]}" -gt 0 ]; then
    log "Failed: ${failed[*]}"
    exit 1
fi
log "All done: ${models[*]}"
