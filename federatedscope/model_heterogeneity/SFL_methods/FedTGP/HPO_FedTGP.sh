set -e
cd ../../../ #
# basic configuration
pass_round=0 # Used when HPO is interrupted: if you don't want to skip certain hyperparameter combinations, make sure pass_round is 0

gpu=0
result_folder_name=FedTGP_2024_01_22
global_eval=False
local_eval_whole_test_dataset=True


method=FedTGP
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}

# common hyperparameters
dataset=('cora' 'citeseer' 'pubmed')
total_client=(3 5 10)
local_update_step=(1 4 10)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.05 0.1 0.25)
total_round=200
patience=50
momentum=0.9
freq=1


# FedTGP-specific parameters
lam=(0.1 1.0 10)
# Define function for model training
cnt=0
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
    eval.freq ${freq} \
    FedTGP.lam ${5}
}

# Loop over parameters for HPO
for data in "${dataset[@]}"; do               # dataset
  for client_num in "${total_client[@]}"; do  # total number of clients
    for lr in "${lrs[@]}"; do                 # local learning rate
      for ls in "${local_update_step[@]}"; do # total number of local update steps
        for lam in "${lam[@]}"; do            # prototype weight
          for s in "${seed[@]}"; do           # seed
            let cnt+=1
            if [ "$cnt" -lt $pass_round ]; then
              continue
            fi
            main_cfg=$script_floder"/"$method"_on_"$data".yaml"
            client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
            exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
            train_model "$client_num" "$s" "$ls" "$lr" "$lam"
          done
        done
      done
    done
  done
done
