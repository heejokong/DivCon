today=`date +%T`
echo $today
seed=1
trainer='dac'
gpu_id=$1

# CIFAR100_TRAIN
dataset=cifar100
num_classes=20
# for n_labels in 5 10 25
for n_labels in 10 25
do
    num_labels=$((n_labels * num_classes))
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --c config/openset_cv/${trainer}/${trainer}_${dataset}_${num_classes}_${num_labels}_${seed}.yaml
done
