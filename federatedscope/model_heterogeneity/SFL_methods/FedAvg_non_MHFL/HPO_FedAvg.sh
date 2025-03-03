set -e
cd ../../../ #到federatedscope目录

# basic configuration
gpu=0
result_folder_name=fedavg_gloabl_eval_hpo_0902
global_eval=False

script_floder="model_heterogeneity/SFL_methods/FedAvg_non_MHFL"
result_floder=model_heterogeneity/result/${result_folder_name}

# common hyperparameters
dataset=('pubmed')
total_client=(3 5 10)
local_update_step=(1 4 16)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.01 0.05 0.25)
total_round=200
patience=50
momentum=0.9
freq=1

# Local-specific parameters
# pass

# Define function for model training
train_model() {
  python main.py --cfg ${main_cfg} \
    federate.client_num ${1} \
    federate.make_global_eval ${global_eval} \
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
          main_cfg=$script_floder"/FedAvg_on_"$data".yaml"
          exp_name="SFL_HPO_fedavg_on_"$data"_"$client_num"_clients"
          train_model "$client_num" "$s" "$ls" "$lr"
        done
      done
    done
  done
done
