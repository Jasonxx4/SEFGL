cd ../../../
python main.py \
  --cfg model_heterogeneity/methods/FML/FML_on_cifar10.yaml \
  --client_cfg model_heterogeneity/model_settings/model_setting_CV_resnet18_low_heterogeneity.yaml
#

#HPO CODE
#CV_low
screen -S FML_hpo_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 2 cifar10 CV_low hpo_0824 False 13

screen -S FML_hpo_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 2 svhn CV_low hpo_0824 False 13

screen -S FML_hpo_low_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 2 office_caltech CV_low hpo_0824 False

#CV_high
screen -S FML_hpo_high_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 3 cifar10 CV_high hpo_0824 False 7

screen -S FML_hpo_high_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 3 svhn CV_high hpo_0824 False 6

screen -S FML_hpo_high_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 3 office_caltech CV_high hpo_0824 False

#CV_low_resnet
screen -S FML_hpo_cv_resnet18_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 2 cifar10 CV_resnet18_low hpo_0824 False 15

screen -S FML_hpo_cv_resnet18_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 2 svhn CV_resnet18_low hpo_0824 False 14

screen -S FML_hpo_cv_resnet18_low_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FML/
bash HPO_FML.sh 4 office_caltech CV_resnet18_low hpo_0824 False