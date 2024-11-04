#!/usr/bin/env bash

cd /afs/cern.ch/work/z/zhangr/HPO/FCG/ML/FastCaloChallenge/training
source /eos/user/z/zhangr/Miniforge3/setup.sh

echo $@

task=$1
relevant_layer=$2
input=$3
output=$4
loading=$5

model=`echo $output | cut -d '_' -f 1`
config_mask=`echo $output | cut -d '_' -f 2-1000`
config_mask=`echo $config_mask | cut -d '.' -f 1`
config=`echo $config_mask | cut -d '-' -f 1`
mask=`echo $config_mask | cut -d '-' -f 2 | cut -d 'M' -f 2`
prep=`echo $config_mask | cut -d '-' -f 3 | cut -d 'P' -f 2`
label_scheme=`echo $config_mask | cut -d '-' -f 4 | cut -d 'L' -f 2`
split_energy=`echo $config_mask | cut -d '-' -f 5 | cut -d 'S' -f 2`
relevant_layer=$(echo "${relevant_layer//_/ }")
max_iter=400000

echo input=$input
echo output=$output
echo mask=$mask
echo prep=$prep
echo loading=$loading
echo label_scheme=$label_scheme
echo split_energy=$split_energy
echo relevant_layer=$relevant_layer
echo max_iter=$max_iter

if [[ $mask == ?(n)+([0-9]) ]]; then
    version='v2'
    addition="--mask=${mask//n/-}"
#if [[ $mask == ?(n)+([0-9]) ]]; then
#    version='v3'
#    train_addition="--mask=${mask//n/-} --add_noise"
else
    version='v1'
    train_addition=""
fi

if [[ ! -z "$prep" ]]; then
    train_addition="$train_addition -p $prep"
    evaluate_addition="$evaluate_addition -p $prep"
fi

if [[ ! -z "$loading" ]]; then
    train_addition="$train_addition $loading"
    evaluate_addition="$evaluate_addition $loading"
fi

if [[ ! -z "$label_scheme" ]]; then
    train_addition="$train_addition --label_scheme $label_scheme"
fi

if [[ ! -z "$split_energy" ]]; then
    train_addition="$train_addition --split_energy_position $split_energy"
    evaluate_addition="$evaluate_addition --split_energy_position $split_energy"
fi

ds=`echo $input | grep -oP '(?<=input/dataset).'`
if [[ "$ds" = "2" ]]; then
    evaluate_addition="$evaluate_addition --normalise"
fi

if [[ ${task} == *'train'* ]]; then
    #command="python train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${output} -c ../config/config_${config}.json ${train_addition}"
    command="python train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${output} -c ../config/config_${config}.json ${train_addition} --relevant_layer ${relevant_layer} --max_iter ${max_iter}"
else
    command="python evaluate.py -i ${input} -t ../output/dataset${ds}/${version}/${output} --relevant_layer ${relevant_layer} --checkpoint ${evaluate_addition}"
fi
echo $command
eval $command
cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds
