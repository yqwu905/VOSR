#!/usr/bin/env bash
# Train an ablation config with train_vosr_ablation.py, then restore every test set
# with inference_vosr_ablation.py from the final export of that same run.
#
#   bash train_and_infer.sh --config configs/ablations/hourglass.yml \
#       --testset preset/datasets/inp_data --testset screensr=/data/ScreenSR/LR \
#       [--nproc 8] [--resume <run>/checkpoint-XXXXXXXX] [-- <inference_vosr_ablation.py options>]
#
# Results go to <run>/inference/<name>/, where <run> is the directory this training
# wrote (output_dir/YYYYmmdd-HHMMSS, or the resumed checkpoint's run directory).
# Everything printed is also appended to <run>/train_and_infer.log.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash train_and_infer.sh --config CONFIG --testset [NAME=]PATH [--testset ...]
                               [--nproc N] [--resume CHECKPOINT] [-- INFERENCE_OPTIONS...]

  --config CONFIG      Ablation config for train_vosr_ablation.py (required).
  --testset [NAME=]PATH
                       LQ image folder (or single image) to restore after training.
                       Repeat for several test sets. NAME defaults to the folder name
                       (file name without extension); outputs go to <run>/inference/NAME/.
  --nproc N            Processes per node for training and inference (default 8).
  --resume CHECKPOINT  Passed to the trainer; the run continues in the checkpoint's directory.
  -- OPTIONS...        Everything after -- is passed to inference_vosr_ablation.py,
                       e.g. -- --tile-size 1024 --upscale 4.

Set TORCHRUN to use another launcher command (default: torchrun).
EOF
}

die() {
    echo "train_and_infer.sh: $*" >&2
    exit 2
}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
torchrun=${TORCHRUN:-torchrun}
config='' resume='' nproc=8
names=() inputs=() counts=() infer_args=()

while (($#)); do
    case $1 in
        --config|--resume|--testset|--nproc)
            (($# >= 2)) || die "$1 needs a value"
            ;;
    esac
    case $1 in
        --config) config=$2; shift 2 ;;
        --resume) resume=$2; shift 2 ;;
        --nproc) nproc=$2; shift 2 ;;
        --testset)
            if [[ $2 =~ ^([A-Za-z0-9._-]+)=(.+)$ ]]; then
                name=${BASH_REMATCH[1]} path=${BASH_REMATCH[2]}
            else
                path=$2 name=$(basename -- "${2%/}")
                [[ -d $path ]] || name=${name%.*}
            fi
            names+=("$name") inputs+=("$path")
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        --) shift; infer_args=("$@"); break ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done

# Check everything before training, so a typo does not surface only after the run.
[[ -n $config ]] || { usage >&2; die "--config is required"; }
[[ -f $config ]] || die "config not found: $config"
((${#names[@]})) || { usage >&2; die "at least one --testset is required"; }
[[ $nproc =~ ^[1-9][0-9]*$ ]] || die "--nproc must be a positive integer, got: $nproc"
[[ -z $resume || -d $resume ]] || die "resume checkpoint not found: $resume"
declare -A seen=()
shopt -s nullglob nocaseglob
for i in "${!names[@]}"; do
    name=${names[i]} path=${inputs[i]}
    [[ -n $name && $name != . && $name != .. ]] || die "cannot name test set '$path' from its path; use NAME=PATH"
    [[ -z ${seen[$name]+x} ]] || die "test set name '$name' is used twice; name them with NAME=PATH"
    seen[$name]=1
    if [[ -d $path ]]; then
        # The extensions inference_vosr_ablation.py reads.
        images=("$path"/*.{png,jpg,jpeg,webp,bmp,tif,tiff})
        ((${#images[@]})) || die "no images in test set '$name': $path"
        counts+=("${#images[@]}")
    elif [[ -f $path ]]; then
        counts+=(1)
    else
        die "test set '$name' not found: $path"
    fi
done
shopt -u nullglob nocaseglob

run_dir_file=$(mktemp)
log=$(mktemp "${TMPDIR:-/tmp}/train_and_infer.XXXXXX")

run() {
    train=("$torchrun" --standalone --nproc_per_node="$nproc" "$repo/train_vosr_ablation.py"
           --config "$config" --run-dir-file "$run_dir_file")
    [[ -z $resume ]] || train+=(--resume "$resume")
    echo "[train] ${train[*]}"
    "${train[@]}"

    run_dir=$(<"$run_dir_file")
    [[ -n $run_dir ]] || die "the trainer did not report its run directory"
    export_dir=$run_dir/export
    [[ -f $export_dir/model.json ]] || die "no final export in $export_dir"
    echo "[train] finished, export: $export_dir"

    failed=() summary=()
    for i in "${!names[@]}"; do
        output=$run_dir/inference/${names[i]}
        # More ranks than images would only load the model and exit.
        procs=$((counts[i] < nproc ? counts[i] : nproc))
        infer=("$torchrun" --standalone --nproc_per_node="$procs" "$repo/inference_vosr_ablation.py"
               --export "$export_dir" --input "${inputs[i]}" --output "$output"
               ${infer_args[@]+"${infer_args[@]}"})
        echo "[infer] ${names[i]}: ${infer[*]}"
        if "${infer[@]}"; then
            written=$(find "$output" -maxdepth 1 -name '*.png' | wc -l)
            echo "[infer] ${names[i]} done: $output"
            summary+=("${names[i]}: $written/${counts[i]} images in $output")
        else
            failed+=("${names[i]}")
            echo "[infer] ${names[i]} failed" >&2
            summary+=("${names[i]}: FAILED, see the [infer] ${names[i]} lines above")
        fi
    done

    echo "==== Summary ===="
    echo "Run directory: $run_dir"
    printf '  %s\n' "${summary[@]}"
    if ((${#failed[@]})); then
        echo "Inference failed for: ${failed[*]}" >&2
        exit 1
    fi
}

# Move the log into the run directory once training has created it, also after a failure.
finish() {
    local status=$? run_dir
    run_dir=$(<"$run_dir_file")
    rm -f "$run_dir_file"
    if [[ -n $run_dir && -d $run_dir ]] && cat "$log" >> "$run_dir/train_and_infer.log"; then
        rm -f "$log"
        echo "Log: $run_dir/train_and_infer.log"
    else
        echo "Log: $log"
    fi
    exit "$status"
}
trap finish EXIT
# Unbuffered Python output keeps the terminal live behind tee.
export PYTHONUNBUFFERED=1
run 2>&1 | tee "$log"
