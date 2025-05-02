today=`date +%T`
echo $today
seed=1
trainer='dac'
gpu_id=$1

# IMAGENET30_TRAIN
dataset=in30
# for percent in p1 p5
for percent in p1
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c config/openset_cv/${trainer}/${trainer}_in30_${percent}_${seed}.yaml
done