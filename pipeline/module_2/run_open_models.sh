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
#     bash pipeline/module_2/run_open_models.sh --gpus 1 internvl3.5-38b
#
# --gpus overrides config.MODELS' n_gpus, for a pod whose single card holds a
# model the config splits across two. --n and --illusion are passed through
# to local_vlm.py, so a short timed run can precede the full ones, and
# --sampled generates every answer rather than drawing them:
#
#     bash pipeline/module_2/run_open_models.sh --pilot --n 100 internvl3.5-2b
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

# The pilot proved this combination: vLLM 0.29.0 with torch 2.13.0+cu132,
# on a driver supporting CUDA 13.0. Both model families answered every
# stimulus, so the full runs pin it. run_info.json and environment.txt
# record the installed version either way; override with VLLM_SPEC.
VLLM_SPEC="${VLLM_SPEC:-vllm==0.29.0}"

# vLLM and PyTorch must be built for the CUDA version the pod's driver
# supports. Plain `pip install vllm` takes the newest build, which on a
# CUDA 12.8 pod fails at engine startup with "The NVIDIA driver on your system
# is too old". uv picks the matching build; "auto" reads the driver, and
# TORCH_BACKEND=cu128 (etc.) forces one.
TORCH_BACKEND="${TORCH_BACKEND:-auto}"

cd "$(dirname "$0")/../.."
REPO_ROOT="$(pwd)"

pilot=0
allow_dirty=0
skip_install=0
gpus=""
sampled=0
n=""
illusion=""
models=()
while [ "$#" -gt 0 ]; do
    arg="$1"
    case "$arg" in
        --gpus) shift; gpus="$1" ;;
        --sampled) sampled=1 ;;
        --n) shift; n="$1" ;;
        --illusion) shift; illusion="$1" ;;
        --pilot) pilot=1 ;;
        --allow-dirty) allow_dirty=1 ;;
        --skip-install) skip_install=1 ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        -*) echo "Unknown option: $arg" >&2; exit 2 ;;
        *) models+=("$arg") ;;
    esac
    shift
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
# The published vLLM wheels are built against CUDA 13: on a pod whose driver
# only supports CUDA 12.x they fail either in torch ("The NVIDIA driver on your
# system is too old") or in vLLM's own extension ("libcudart.so.13: cannot open
# shared object file"). Installing a CUDA 12 build of torch does not help,
# because the vLLM extension is the part that needs CUDA 13. Deploy a pod whose
# driver supports CUDA 13 instead; on RunPod, filter the GPU list by CUDA
# version. REQUIRED_CUDA=12 skips this guard if a future wheel allows it.
driver_cuda="$(nvidia-smi -q 2>/dev/null | awk -F': *' '/CUDA Version/ {print $2; exit}')"
log "Driver supports CUDA ${driver_cuda:-unknown}"
case "${driver_cuda%%.*}" in
    ''|*[!0-9]*) echo "WARNING: could not read the driver's CUDA version." ;;
    *)
        if [ "${driver_cuda%%.*}" -lt "${REQUIRED_CUDA:-13}" ]; then
            echo "This pod's driver supports CUDA $driver_cuda, but the vLLM wheels need CUDA ${REQUIRED_CUDA:-13}." >&2
            echo "Deploy a pod whose driver supports CUDA ${REQUIRED_CUDA:-13} (filter the GPU list by CUDA version)." >&2
            exit 1
        fi
        ;;
esac

if [ "$skip_install" -eq 0 ]; then
    log "Installing $VLLM_SPEC for CUDA backend '$TORCH_BACKEND', plus pillow and pandas"
    python -m pip install --quiet uv
    # --break-system-packages: the pod image marks its Python as externally
    # managed (Debian), which uv refuses to touch without it. pip on these
    # images is already configured to allow it.
    uv pip install --system --break-system-packages --quiet \
        --torch-backend="$TORCH_BACKEND" "$VLLM_SPEC"
    python -m pip install --quiet pillow pandas
fi
python -c "
import torch, vllm
print('vLLM', vllm.__version__, '| torch', torch.__version__, '| built for CUDA', torch.version.cuda)
"
# FlashInfer compiles its kernels on first use and takes the target
# architecture from TORCH_CUDA_ARCH_LIST, which the pod images pin to the
# architectures they were built for. On a card newer than that list,
# FlashInfer reads an old architecture and refuses to build at all
# ("FlashInfer requires GPUs with sm75 or higher"), killing the engine
# during warm-up. Point the list at the card actually present.
detected_arch="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))' 2>/dev/null || true)"
if [ -n "$detected_arch" ]; then
    export TORCH_CUDA_ARCH_LIST="$detected_arch"
    log "Kernels will be built for CUDA capability $detected_arch"
fi

# vLLM's own top-k/top-p sampler needs no compilation, and with top_p = 1
# and no top_k it draws from the same distribution as FlashInfer's, so the
# sampler is one fewer architecture-specific kernel to build per session.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

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
    [ -n "$gpus" ] && need="$gpus"
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
        echo "--- kernel and sampler settings"
        env | grep -E '^(VLLM_|TORCH_CUDA_ARCH_LIST=|HF_HOME=)' | grep -viE 'token|key|secret|password' | sort || true
        echo
        echo "--- pip freeze"
        python -m pip freeze
    } > "$out/environment.txt"

    log "Running $m${flag:+ (pilot)}"
    if python -m pipeline.module_2.local_vlm --model "$m" $flag ${gpus:+--gpus "$gpus"} ${n:+--n "$n"} ${illusion:+--illusion "$illusion"} $([ "$sampled" -eq 1 ] && echo --sampled) 2>&1 | tee "$out/run.log"; then
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
