set -e
cd ../../../ #到federatedscope目录

# Configuration
gpu=$1
dataset=$2 #cifar10,svhn,office_caltech
task=$3    #CV_Low CV_High CV_resnet18
folder_name=$4
local_eval_whole_test_dataset=$5 #别忘了在train_model()加上这个参数

# WandB setup
wandb_use=False
wandb_name_user=niudaidai
wandb_online_track=False
wandb_client_train_info=True
wandb_name_project="HPO_"$method"_on_"$dataset"_for_"$task

# Method setup
method=FedProto
client_file="model_heterogeneity/model_settings/model_setting_"$task"_heterogeneity.yaml"
result_floder=model_heterogeneity/result/${folder_name}
script_floder="model_heterogeneity/methods/"${method}
main_cfg=${script_floder}"/${method}""_on_"${dataset}".yaml"
exp_name="HPO_"$method"_on_"$dataset"_"$task

# Hyperparameters
if [ "$dataset" == "svhn" ] && [ "$task" == 'CV_high' ]; then
  lrs=(0.001 0.0001)
else
  lrs=(0.001)
fi
local_update_step=(1)
optimizer=('Adam')
seed=(0)
total_round=300
patience=5
momentum=0.9
freq=10

# FedProto-specific parameters
proto_weight=(0.1 1.0 10)

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
    fedproto.proto_weight ${5} \
    fedproto.show_verbose False \
    MHFL.task ${task} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset}
}

# Training parameters based on the dataset
declare -A lda_alpha_map=(
  ["cifar10"]="1.0 0.5 0.1"
  ["svhn"]="1.0 0.5 0.1"
  ["office_caltech"]="1.0 0.5 0.1"
)
lda_alpha=(${lda_alpha_map[$dataset]})

# Loop over parameters for HPO
for alpha in "${lda_alpha[@]}"; do
  for opt in "${optimizer[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
        for weight in "${proto_weight[@]}"; do
          for s in "${seed[@]}"; do
            splitter_args="data.splitter_args ""[{'alpha':${alpha}}]"
            train_model "$s" "$ls" "$opt" "$lr" "$weight"
          done
        done
      done
    done
  done
done
