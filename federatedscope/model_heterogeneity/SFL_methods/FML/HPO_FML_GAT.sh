set -e
cd ../../../ #到federatedscope目录

# basic configuration
gpu=0
meme_model=gat
result_folder_name="FML_"${meme_model}"_HPO_0903"
local_eval_whole_test_dataset=False
pass_round=716

method=FML
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}

# common hyperparameters
dataset=('cora' 'citeseer' 'pubmed')
total_client=(3 5 10)
local_update_step=(1 4 16)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.01 0.05 0.25)
total_round=200
patience=50
momentum=0.9
freq=1

# FML-specific parameters
fml_alpha=(0.3 0.5 1.0)
fml_beta=(0.3 0.5 1.0)

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
    fml.alpha ${5} \
    fml.beta ${6} \
    fml.meme_model.type ${meme_model} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset}
}
cnt=0
# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
        for alpha in "${fml_alpha[@]}"; do
          for beta in "${fml_beta[@]}"; do
            for s in "${seed[@]}"; do
              let cnt+=1
              if [ "$cnt" -lt $pass_round ]; then
                continue
              fi
              main_cfg=$script_floder"/"$method"_on_"$data".yaml"
              client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
              exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
              train_model "$client_num" "$s" "$ls" "$lr" "$alpha" "$beta"
            done
          done
        done
      done
    done
  done
done
