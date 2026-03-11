#!/usr/bin/env bash

cd /home/HPC/corchiahpc/ruiGAN/FastCaloChallengeQuantumOpenData/training

echo "$@"

task=$1
input=$2
output=$3
loading=$4

model=$(echo "$output" | cut -d '_' -f 1)
config_mask=$(echo "$output" | cut -d '_' -f 2-1000)
config_mask=$(echo "$config_mask" | cut -d '.' -f 1)
config=$(echo "$config_mask" | cut -d '-' -f 1)
mask=$(echo "$config_mask" | cut -d '-' -f 2 | cut -d 'M' -f 2)
prep=$(echo "$config_mask" | cut -d '-' -f 3 | cut -d 'P' -f 2)
label_scheme=$(echo "$config_mask" | cut -d '-' -f 4 | cut -d 'L' -f 2)
split_energy=$(echo "$config_mask" | cut -d '-' -f 5 | cut -d 'S' -f 2)

echo input="$input"
echo output="$output"
echo model="$model"
echo config="$config"
echo mask="$mask"
echo prep="$prep"
echo loading="$loading"
echo label_scheme="$label_scheme"
echo split_energy="$split_energy"

if [[ $mask == ?(n)+([0-9]) ]]; then
    version='v2'
    train_addition="--mask=${mask//n/-}"
else
    version='v1'
    train_addition=""
fi

if [[ -n "$prep" ]]; then
    train_addition="$train_addition -p $prep"
    evaluate_addition="$evaluate_addition -p $prep"
fi

if [[ -n "$loading" ]]; then
    train_addition="$train_addition $loading"
    evaluate_addition="$evaluate_addition $loading"
fi

if [[ -n "$label_scheme" ]]; then
    train_addition="$train_addition --label_scheme $label_scheme"
fi

if [[ -n "$split_energy" ]]; then
    train_addition="$train_addition --split_energy_position $split_energy"
    evaluate_addition="$evaluate_addition --split_energy_position $split_energy"
fi

ds=$(echo "$input" | grep -oP '(?<=input/dataset).')
base_output="../output/dataset${ds}/${version}/${output}"

if [[ "$ds" = "2" ]]; then
    evaluate_addition="$evaluate_addition --normalise"
fi

# Dedicated standalone qINN flow (separate from GAN chain).
# Trigger rules (priority order):
#  1) FASTCALO_QINN_LEGACY_GAN=1 -> force legacy GAN path
#  2) FASTCALO_QINN_STANDALONE=1 -> force standalone qINN path
#  3) task contains "qinn" OR model token is "qinnstandalone" OR config token contains "qinnstandalone"
is_qinn_standalone=0
if [[ "${FASTCALO_QINN_LEGACY_GAN:-0}" == "1" ]]; then
    is_qinn_standalone=0
elif [[ "${FASTCALO_QINN_STANDALONE:-0}" == "1" ]]; then
    is_qinn_standalone=1
elif [[ "$task" == *"qinn"* ]] || [[ "${model,,}" == "qinnstandalone" ]] || [[ "${config,,}" == *"qinnstandalone"* ]]; then
    is_qinn_standalone=1
fi

if [[ $is_qinn_standalone -eq 1 ]]; then
    if [[ ${task} == *'train'* ]]; then
        command="python train_qinn.py -i ${input} -o ${base_output} -c ../config/config_${config}.json --epochs 200 --checkpoint_every 10 --progress_every 1"
    else
        # If checkpoint is not provided in arg4, pick latest qinn checkpoint automatically.
        if [[ -n "$loading" && -f "$loading" ]]; then
            ckpt_path="$loading"
        else
            ckpt_path=$(ls -1 ${base_output}/checkpoints/qinn_module_state-*.pt 2>/dev/null | sort -V | tail -n 1)
        fi
        if [[ -z "$ckpt_path" ]]; then
            echo "[ERROR] No qINN checkpoint found in ${base_output}/checkpoints"
            exit 1
        fi
        command="python evaluate_qinn.py -i ${input} --checkpoint ${ckpt_path} -o ${base_output}/evaluate_qinn -c ../config/config_${config}.json"
    fi
else
    # Backward-compatible GAN/qGAN flow
    if [[ ${task} == *'train'* ]]; then
        # keep --quantum for legacy qGAN flows; add progress logs every 100 iterations
        command="python train.py -i ${input} -m ${model} -o ${base_output} -c ../config/config_${config}.json ${train_addition} --max_iter 1000000 --progress_interval 100 --quantum"
    else
        command="python evaluate.py -i ${input} -t ${base_output} --checkpoint ${evaluate_addition} --debug --save_h5 --quantum"
    fi
fi

echo "$command"
eval "$command"
cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds
