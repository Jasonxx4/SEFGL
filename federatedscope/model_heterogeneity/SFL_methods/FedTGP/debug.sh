cd ../../../
python main.py \
--cfg model_heterogeneity/SFL_methods/FedTGP/FedTGP_on_cora.yaml \
--client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml \
data.local_eval_whole_test_dataset True


--cfg
model_heterogeneity/SFL_methods/FedTGP/FedTGP_on_cora.yaml
--client_cfg
model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml
data.local_eval_whole_test_dataset
True



vis_embedding
True
show_client_best_individual
True
federate.total_round_num
2
save_history_result_per_client
True