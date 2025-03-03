cd ../../../
#python main.py \
#--cfg model_heterogeneity/SFL_methods/POI/POI_on_cora.yaml \
#--client_cfg model_heterogeneity/model_settings/model_setting_GNNs_3_clients_without_additional_fc.yaml
#
#
##POIV1
#python main.py
#--cfg model_heterogeneity/SFL_methods/POI/POIV1_on_cora.yaml
#--client_cfg model_heterogeneity/model_settings/model_setting_GNNs_3_clients_without_additional_fc.yaml
#
#
##POIV2
#python main.py
#--cfg model_heterogeneity/SFL_methods/POI/POIV2_on_cora.yaml
#--client_cfg model_heterogeneity/model_settings/model_setting_GNNs_3_clients_without_additional_fc.yaml

#POIV3
# 3client
python main.py \
--cfg model_heterogeneity/SFL_methods/POI/POIV3_on_cora.yaml \
--client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml

#5 client
--cfg model_heterogeneity/SFL_methods/POI/POIV3_on_cora.yaml
--client_cfg model_heterogeneity/model_settings/5_Heterogeneous_GNNs.yaml
federate.client_num 5
#POIV4
python main.py \
--cfg model_heterogeneity/SFL_methods/POI/POIV4_on_cora.yaml \
--client_cfg model_heterogeneity/model_settings/3_Heterogeneous_GNNs.yaml