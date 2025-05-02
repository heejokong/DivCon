today=`date +%T`
echo $today
seed=1
trainer='dac'
gpu_id=$1

# SEMI_INAT_2021_TRAIN
dataset=semi_inat
CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c config/openset_cv/${trainer}/${trainer}_semi_inat_${seed}.yaml
