#HPO Script
#CV_low
screen -S FedMD_hpo_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 4 cifar10 CV_low hpo_0828 False

screen -S FedMD_hpo_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 4 svhn CV_low hpo_0828 False

screen -S FedMD_hpo_low_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 4 office_caltech CV_low hpo_0828 False

#CV_high
screen -S FedMD_hpo_high_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 2 cifar10 CV_high hpo_0828 False

screen -S FedMD_hpo_high_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 4 svhn CV_high hpo_0828 False

screen -S FedMD_hpo_high_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 5 office_caltech CV_high hpo_0828 False

#CV_low_resnet
screen -S FedMD_hpo_cv_resnet18_low_cifar10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 5 cifar10 CV_resnet18_low hpo_0828 False

screen -S FedMD_hpo_cv_resnet18_low_svhn
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 5 svhn CV_resnet18_low hpo_0828 False

screen -S FedMD_hpo_cv_resnet18_low_office_10
conda activate mhfl
cd ../../data/zhl2021/FedMM/federatedscope/model_heterogeneity/methods/FedMD/
bash HPO_FedMD.sh 5 office_caltech CV_resnet18_low hpo_0828 False