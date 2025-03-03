set -e
cd ../../../ #到federatedscope目录
# basic configuration
# cd /data/yhp2022/FGPL/federatedscope/model_heterogeneity/SFL_methods/FGPL
gpu=3
result_folder_name=FGPL
global_eval=False
local_eval_whole_test_dataset=True
method=FGPL
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}
# common hyperparameters
dataset=('computers')
total_client=(10)
local_update_step=(5 10)
optimizer='SGD'
lrs=(0.05 0.1 0.25)
total_round=200
patience=60
momentum=0.9
freq=1
pass_round=0
beta=(100)
# Local-specific parameters
proto_weight=(0.1 0.2)
mu=(0.3 0.5 0.8)
imb_ratio=(100 1 20)
lamda=(0.5)
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
    fedproto.proto_weight ${5} \
    graphsha.beta ${6} \
    fedproto.mu ${7} \
    graphsha.imb_ratio ${8} \
    fedproto.lamda ${9}
}

# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    for s in "${seed[@]}"; do
      for lr in "${lrs[@]}"; do
        for ls in "${local_update_step[@]}"; do
          for weight in "${proto_weight[@]}"; do
            for beta in "${beta[@]}"; do
              for mu in "${mu[@]}"; do
                for imb_ratio in "${imb_ratio[@]}"; do
                  for lamda in "${lamda[@]}"; do
                    let cnt+=1
                    if [ "$cnt" -lt $pass_round ]; then
                      continue
                    fi
                    main_cfg=$script_floder"/"$method"_on_"$data".yaml"
                    client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
                    exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
                    train_model "$client_num" "$s" "$ls" "$lr" "$weight" "$beta" "$mu" "$imb_ratio" "$lamda"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
