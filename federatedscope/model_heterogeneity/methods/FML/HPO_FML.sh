set -e
cd ../../../ #到federatedscope目录

# Configuration
gpu=$1
dataset=$2 #cifar10,svhn,office_caltech
task=$3    #CV_Low CV_High CV_resnet18
result_folder_name=$4
local_eval_whole_test_dataset=$5  #别忘了在train_model()加上这个参数

if [ $# -ge 6 ]; then
    pass_round="$6"
else
    # 如果没有提供第5个参数，则设置默认值
    pass_round=0
fi



# Method setup
method=FML
client_file="model_heterogeneity/model_settings/model_setting_"$task"_heterogeneity.yaml"
result_floder=model_heterogeneity/result/${result_folder_name}
script_floder="model_heterogeneity/methods/"${method}
main_cfg=${script_floder}"/${method}""_on_"${dataset}".yaml"
exp_name="HPO_"$method"_on_"$dataset"_"$task

# WandB setup
wandb_use=False
wandb_name_user=niudaidai
wandb_online_track=False
wandb_client_train_info=True
wandb_name_project="HPO_"$method"_on_"$dataset"_for_"$task

# Hyperparameters
local_update_step=(1)

# 0827：对于CIFAR,SVHN来说，local_update_step为1和5没有显著差异；为了节省时间，将local_update_step指定为1
# 似乎只有在CV_high并且数据集是svhn的时候，lr=0.0001的效果才会比lr=0.001好。同样为了节省时间，只对这种情况下对学习率搜索，其他情况指定lr=0.001
lrs=(0.001)
if [ "$dataset" == "svhn" ] && [ "$task" == 'CV_high' ]; then
  lrs=(0.001 0.0001)
fi

optimizer=('Adam')
seed=(0)
total_round=300
patience=50
momentum=0.9
freq=1

# FML-specific parameters
fml_alpha=(0.3 0.5 1.0)
fml_beta=(0.3 0.5 1.0)


# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} --client_cfg ${client_file} \
    federate.total_round_num ${total_round} \
    early_stop.patience ${patience} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    seed ${1} \
    train.local_update_steps ${2} \
    train.optimizer.type ${3} \
    train.optimizer.lr ${4} \
    train.optimizer.momentum ${momentum} \
    device ${gpu} \
    ${splitter_args} \
    wandb.use ${wandb_use} \
    wandb.name_user ${wandb_name_user} \
    wandb.name_project ${wandb_name_project} \
    wandb.online_track ${wandb_online_track} \
    wandb.client_train_info ${wandb_client_train_info} \
    eval.freq ${freq} \
    fml.alpha ${5} \
    fml.beta ${6} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset}
}

# Training parameters based on the dataset
declare -A lda_alpha_map=(
  ["cifar10"]="1.0 0.5 0.1"
  ["svhn"]="1.0 0.5 0.1"
  ["office_caltech"]="1.0 0.5 0.1"
)
lda_alpha=(${lda_alpha_map[$dataset]})

cnt=0
# Loop over parameters for HPO
for alpha in "${lda_alpha[@]}"; do
  for opt in "${optimizer[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
        for a in "${fml_alpha[@]}"; do
          for b in "${fml_beta[@]}"; do
            for s in "${seed[@]}"; do
              let cnt+=1
              if [ "$cnt" -lt $pass_round ]; then
                continue
              fi
              splitter_args="data.splitter_args ""[{'alpha':${alpha}}]"
              train_model "$s" "$ls" "$opt" "$lr" "$a" "$b"
            done
          done
        done
      done
    done
  done
done
