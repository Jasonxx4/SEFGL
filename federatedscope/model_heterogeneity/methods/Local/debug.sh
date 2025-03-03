cd ../../../
python main.py \
  --cfg model_heterogeneity/methods/Local/Local_on_cifar10.yaml \
  --client_cfg model_heterogeneity/model_settings/model_setting_CV_resnet18_low_heterogeneity.yaml

#HPO CODE
#CV_low
screen -S local_hpo_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 0 cifar10 CV_low hpo_0824 False

screen -S local_hpo_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 0 svhn CV_low hpo_0824 False

screen -S local_hpo_low_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 0 office_caltech CV_low hpo_0824 False

#CV_high
screen -S local_hpo_high_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 1 cifar10 CV_high hpo_0824 False

screen -S local_hpo_high_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 1 svhn CV_high hpo_0824 False

screen -S local_hpo_high_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 0 office_caltech CV_high hpo_0824 False

#CV_low_resnet
screen -S local_hpo_cv_resnet18_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 2 cifar10 CV_resnet18_low hpo_0824 False

screen -S local_hpo_cv_resnet18_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 2 svhn CV_resnet18_low hpo_0824 False

screen -S local_hpo_cv_resnet18_low_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/Local/
bash HPO_Local.sh 3 office_caltech CV_resnet18_low hpo_0824 False