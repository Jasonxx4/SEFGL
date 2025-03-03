set -e
cd ../../../ #到federatedscope目录

# basic configuration
gpu=0
result_folder_name=POIV3_HPO_0908
global_eval=False
local_eval_whole_test_dataset=False

pass_round=0

method=POIV3
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}

# common hyperparameters
#dataset=('cora' 'citeseer' 'pubmed')
dataset=('cora')
total_client=(5)
local_update_step=(1 4)
optimizer='SGD'
seed=(0 1 2)
#lrs=(0.01 0.05 0.25)
lrs=(0.05 0.25)
total_round=200
patience=50
momentum=0.9
freq=1

# POIV5-specific parameters
LP_layer=(1 2 3)
LP_alpha=(0.1 0.5 0.9)

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
    poi.LP_layer ${5} \
    poi.LP_alpha ${6}
}

# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
        for lp_k in "${LP_layer[@]}"; do
          for lp_a in "${LP_alpha[@]}"; do
            for s in "${seed[@]}"; do
              let cnt+=1
              if [ "$cnt" -lt $pass_round ]; then
                continue
              fi
              main_cfg=$script_floder"/"$method"_on_"$data".yaml"
              client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
              exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
              train_model "$client_num" "$s" "$ls" "$lr" "$lp_k" "$lp_a"
            done
          done
        done
      done
    done
  done
done
