set -e
cd ../../../ #到federatedscope目录
# basic configuration
# cd /data/yhp2022/FS/federatedscope/model_heterogeneity/SFL_methods/FedProto
gpu=8
result_folder_name=local_sha
global_eval=False
local_eval_whole_test_dataset=True
method=Fed_sha_local
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}
# common hyperparameters
dataset='citeseer'
total_client=(10)
local_update_step=(8 16 24 32)
optimizer='SGD'
seed=(0 1 2)
lrs=(0.01 0.03 0.05 0.1 0.25)
beta=(1 2 5 10 100 1000)
total_round=60
patience=30
momentum=0.9
freq=1
pass_round=0
# Local-specific parameters
lamda=(0.1 0.5 1.0)
warmup=(3 4 5)
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
    fedproto.lamda ${5} \
    graphsha.warmup ${6} \
    graphsha.beta ${7}
}

# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for lr in "${lrs[@]}"; do
      for ls in "${local_update_step[@]}"; do
        for s in "${seed[@]}"; do
          for lamda in "${lamda[@]}"; do
            for warmup in "${warmup[@]}"; do
              for beta in "${beta[@]}"; do
                let cnt+=1
                if [ "$cnt" -lt $pass_round ]; then
                  continue
                fi
                main_cfg=$script_floder"/"$method"_on_"$data".yaml"
                client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
                exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
                train_model "$client_num" "$s" "$ls" "$lr" "$lamda" "$warmup" "$beta"
              done
            done
          done
        done
      done
    done
  done
done
