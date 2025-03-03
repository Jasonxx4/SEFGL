set -e
cd ../../../ #到federatedscope目录
# basic configuration
# cd /data/yhp2022/FGPL/federatedscope/model_heterogeneity/SFL_methods/FGPL
gpu=5
result_folder_name=FGPL
global_eval=False
local_eval_whole_test_dataset=True
method=FGPL
script_floder="model_heterogeneity/SFL_methods/"${method}
result_floder=model_heterogeneity/result/${result_folder_name}
# common hyperparameters
dataset=('computers')
total_client=(5 7 10)
optimizer='SGD'
delta=(0.1 0.3 0.5)
# Define function for model training
cnt=0
train_model() {
  python main.py --cfg ${main_cfg} \--client_cfg ${client_cfg} \
    federate.client_num ${1} \
    federate.make_global_eval ${global_eval} \
    data.local_eval_whole_test_dataset ${local_eval_whole_test_dataset} \
    result_floder ${result_floder} \
    device ${gpu}
}
# Loop over parameters for HPO
for data in "${dataset[@]}"; do
  for client_num in "${total_client[@]}"; do
    main_cfg=$script_floder"/"$method"_on_"$data".yaml"
    client_cfg="model_heterogeneity/model_settings/"$client_num"_Heterogeneous_GNNs.yaml"
    exp_name="SFL_HPO_"$method"_on_"$data"_"$client_num"_clients"
    train_model "$client_num"
  done
done
