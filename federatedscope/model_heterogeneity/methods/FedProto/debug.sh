cd ../../../
python main.py \
--cfg model_heterogeneity/methods/FedProto/FedProto_on_cifar10.yaml \
--client_cfg model_heterogeneity/model_settings/model_setting_CV_low_heterogeneity.yaml


#HPO script
#CV_low
screen -S FedProto_hpo_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 2 cifar10 CV_low hpo_0827 False

screen -S FedProto_hpo_low_svhn器
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 2 svhn CV_low hpo_0827 False

screen -S FedProto_hpo_low_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 2 office_caltech CV_low hpo_0827 False

#CV_high
screen -S FedProto_hpo_high_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 5 cifar10 CV_high hpo_0827 False

screen -S FedProto_hpo_high_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 5 svhn CV_high hpo_0827 False

screen -S FedProto_hpo_high_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 5 office_caltech CV_high hpo_0827 False

#CV_low_resnet
screen -S FedProto_hpo_cv_resnet18_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 4 cifar10 CV_resnet18_low hpo_0827 False

screen -S FedProto_hpo_cv_resnet18_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 4 svhn CV_resnet18_low hpo_0827 False

screen -S FedProto_hpo_cv_resnet18_low_10c
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedProto/
bash HPO_FedProto.sh 4 office_caltech CV_resnet18_low hpo_0827 False