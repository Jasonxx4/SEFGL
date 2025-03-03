set -e
cd ../../../ #到federatedscope目录
# cd /data/yhp2022/FS/federatedscope/model_heterogeneity/SFL_methods/FedAPEN
# basic configuration
gpu=5
method=FedAPEN
global_model=gcn

result_folder_name="FedAPEN_"${global_model}"_HPO_1210_shared_local_model"
local_eval_whole_test_dataset=True

script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}

share_local_model=True
# common hyperparameters
dataset=('cora')
total_client=(5 7 10)
local_update_step=(1 4 16)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.05 0.25 0.1)
total_round=200
patience=50
momentum=0.9
freq=1

# FedAPEN-specific parameters
# PASS

# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} \--client_cfg ${client_cfg} \
    federate.client_num ${1} \
    seed ${2} \
    train.local_update_steps ${3} \
    train.optimizer.lr ${4} \
    federate.total_round_num ${total_round} \
    train.optimizer.type ${optimizer} \
    train.optimizer.momentum ${momentum} \
    device ${gpu} \
    early_stop.patience ${patience} \
    result_floder ${result_floder} \
    exp_name ${exp_name} \
    eval.freq ${freq} \
    MHFL.global_model.type ${global_model} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset} \
    federate.share_local_model ${share_local_model}
}

# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
          for s in "${seed[@]}"; do
            main_cfg=$script_floder"/"$method"_on_"$data".yaml"
            client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
            exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
            train_model "$client_num" "$s" "$ls" "$lr"
          done
      done
    done
  done
done
