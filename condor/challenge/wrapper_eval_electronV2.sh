#!/usr/bin/env bash

cd /afs/cern.ch/work/z/zhangr/FCG/FastCaloChallenge/training
source /afs/cern.ch/work/z/zhangr/HH4b/hh4bStat/scripts/setup.sh

echo $@

task=$1
input=$2
slice=$3
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

echo input=$input
echo output=$output
echo mask=$mask
echo prep=$prep
echo loading=$loading
echo label_scheme=$label_scheme
echo split_energy=$split_energy

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
    evaluate_addition="-p $prep"
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

evaluate_addition="$evaluate_addition --islice $slice"

ds=`echo $input | grep -oP '(?<=input/dataset).'`

if [[ ${task} == *'train'* ]]; then
    command="python train.py -i ${input} -m ${model} -o ../output/dataset${ds}/${version}/${output} -c ../config/config_${config}.json ${train_addition}"
else
    command="python evaluate.py -i ${input} -t ../output/dataset${ds}/${version}/${output} --checkpoint ${evaluate_addition}"
fi
echo $command
$command
cd -
unset mask prep config config_mask model train_addition evaluate_addition loading label_scheme ds
