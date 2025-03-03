set -e
cd ../../../ #到federatedscope目录
# basic configuration
gpu=4
result_folder_name=local_hpo_0904_test_on_whole_graph
global_eval=False
local_eval_whole_test_dataset=True
method=Local
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}
# common hyperparameters
dataset=('computers')
total_client=(7)
local_update_step=(4 16 32)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.01 0.1 0.25)
total_round=200
patience=50
momentum=0.9
freq=1
# Local-specific parameters
# pass
# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} \--client_cfg ${client_cfg} \
    federate.client_num ${1} \
    federate.make_global_eval ${global_eval} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset} \
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
    eval.freq ${freq}
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
