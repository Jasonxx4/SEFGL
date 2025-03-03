cd ../../../
python main.py \
--cfg model_heterogeneity/SFL_methods/FML/FML_on_photo.yaml
--client_cfg model_heterogeneity/model_settings/7_Heterogeneous_GNNs.yaml
data.local_eval_whole_test_dataset True
