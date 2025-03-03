import time

from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config
import logging
import os

logger = logging.getLogger(__name__)


def extend_model_heterogeneous_cfg(cfg):
    '''
    模型异构联邦学习用到的通用参数
    MHFL: Model Heterogeneous Federated Learning
    '''

    cfg.MHFL = CN()
    cfg.MHFL.task = 'CV_low_heterogeneity'
    cfg.MHFL.all_local = False  # 为True则确保

    # ------------------------------------------------------------------------------
    # global shared model realted options (for FML and FedKD)
    # ------------------------------------------------------------------------------
    # TODO: 将FML的 meme_model 替换为 global model
    cfg.MHFL.global_model = CN()
    cfg.MHFL.global_model.type = 'CNN_3layers'
    cfg.MHFL.global_model.hidden = 256
    cfg.MHFL.global_model.dropout = 0.5
    cfg.MHFL.global_model.filter_channels = [64, 64, 64]
    cfg.MHFL.global_model.return_proto = False
    cfg.MHFL.global_model.out_channels = 10
    cfg.MHFL.global_model.use_bn = True
    cfg.MHFL.global_model.LP_alpha = 0.9  # For label propagation algorithm
    cfg.MHFL.global_model.task = 'CV_low'
    cfg.MHFL.global_model.input_shape = [2708, 1433]
    cfg.MHFL.global_model.layer = 2
    cfg.MHFL.global_model.warpFC = False
    cfg.MHFL.global_model.feature_dim = 64
    cfg.MHFL.global_model.num_classes = 7
    cfg.MHFL.public_train = CN()  # 在公共数据集上训练相关的参数
    cfg.MHFL.public_dataset = 'mnist'
    cfg.MHFL.public_len = 5000  # weak or strong
    cfg.MHFL.model_weight_dir = './contrib/model_weight'

    # public training optimizer相关
    cfg.MHFL.public_train.optimizer = CN()
    cfg.MHFL.public_train.optimizer.type = 'Adam'
    cfg.MHFL.public_train.optimizer.lr = 0.001
    cfg.MHFL.public_train.optimizer.weight_decay = 0.
    # cfg.MHFL.public_train.optimizer.momentum = 1e-4

    # Pretrain related option
    cfg.MHFL.pre_training = CN()
    cfg.MHFL.pre_training.public_epochs = 1
    cfg.MHFL.pre_training.private_epochs = 1
    cfg.MHFL.pre_training.public_batch_size = 256
    cfg.MHFL.pre_training.private_batch_size = 256
    cfg.MHFL.pre_training.rePretrain = True
    cfg.MHFL.pre_training.save_model = True  # 是否保存预训练模型

    # dataset realated option
    cfg.data.local_eval_whole_test_dataset = False
    cfg.result_floder = 'model_heterogeneity/result/manual'
    cfg.exp_name = cfg.federate.method + "_on_" + "cfg.data.type_" + str(time.time())

    # model related option
    cfg.model.filter_channels = [64, 64, 64]
    cfg.model.use_bn = True

    cfg.model.warpFC = False  # Whether to add a fully connected layer at the end of the local model
    cfg.model.feature_dim = -1  # When warpFC is True, the last dimension of each local model output will be mapped to the "feature_dim" dimension by nn.AdaptiveAvgPool1d

    cfg.model.return_proto = False  # 用来控制是否在前向传播时输出每个输入对应的表征。FedProto,FedPCL,FedheNN等需要用到
    cfg.model.num_classes = -1  # FederatedScope中原代码中没有这个变量，但是这个变量在创建模型时很常用，故添加

    # other option
    cfg.train.optimizer.momentum = 0.9
    cfg.show_label_distribution = False  # 可视化相关参数
    cfg.show_client_best_individual = True
    cfg.show_detailed_communication_info = False

    cfg.MHFL.add_label_index = False
    cfg.MHFL.emb_file_path = 'model_heterogeneity/embedding'  # Note: if cfg.vis_embedding is true, the code will delete the files in the emb_file_path

    cfg.vis_embedding = False  # 是否保存每一轮的node embedding用以可视化
    cfg.plot_acc_curve = False

    # ------------------------------------------------------------------------------------------------------------------------------
    '''
    Hyperparameters required for each algorithm in hits Benchmark
    '''
    # ---------------------------------------------------------------------- #
    # FedMD: Heterogenous Federated Learning via Model Distillation
    # ---------------------------------------------------------------------- #
    cfg.fedmd = CN()

    # Pre-training steps before starting federated communication
    cfg.fedmd.pre_training = CN()
    cfg.fedmd.pre_training.public_epochs = 1
    cfg.fedmd.pre_training.private_epochs = 1
    cfg.fedmd.pre_training.public_batch_size = 256
    cfg.fedmd.pre_training.private_batch_size = 256
    cfg.fedmd.pre_training.rePretrain = True

    # Communication step
    cfg.fedmd.public_subset_size = 5000

    # Digest step
    cfg.fedmd.digest_epochs = 1
    # Revisit step
    cfg.fedmd.revisit_epochs = 1

    # ---------------------------------------------------------------------- #
    # FedProto related options
    # ---------------------------------------------------------------------- #
    cfg.fgpl = CN()
    cfg.fgpl.proto_weight = 0.1  # weight of proto loss\\
    cfg.fgpl.n_cls = 40
    cfg.fgpl.infoNCET = 0.02
    cfg.fgpl.lamda = 0.5
    cfg.fgpl.gama = 0.9
    cfg.fgpl.mu = 0.7
    cfg.fgpl.imb_ratio = 80
    cfg.fgpl.gdc = 'ppr'
    cfg.fgpl.warmup = 5
    cfg.fgpl.tau = 2
    cfg.fgpl.delta = 0.1
    cfg.fgpl.beta = 100
    cfg.fgpl.show_verbose = False

    cfg.sefgl = CN()
    cfg.sefgl.proto_weight = 0.1  # weight of proto loss\\
    cfg.sefgl.n_cls = 40
    cfg.sefgl.infoNCET = 0.02
    cfg.sefgl.lamda = 0.5
    cfg.sefgl.gama = 0.9
    cfg.sefgl.mu = 0.7
    cfg.sefgl.imb_ratio = 80
    cfg.sefgl.gdc = 'ppr'
    cfg.sefgl.warmup = 5
    cfg.sefgl.tau = 2
    cfg.sefgl.delta = 0.1
    cfg.sefgl.beta = 100
    cfg.sefgl.show_verbose = False

    # Model related options
    cfg.model.stride = [1, 4]
    cfg.model.fedproto_femnist_channel_temp = 18
    cfg.model.pretrain_resnet = False

    # data related options
    cfg.fedproto = CN()
    cfg.fedproto.iid = False
    cfg.fedproto.unequal = False
    cfg.fedproto.ways = 5
    cfg.fedproto.stdev = 2
    cfg.fedproto.shots = 100
    cfg.fedproto.train_shots_max = 110
    cfg.fedproto.test_shots = 15
    cfg.fedproto.proto_weight= 0.1
    cfg.fedproto.n_cls = 7
    cfg.fedproto.infoNCET = 0.02
    cfg.fedproto.lamda = 0.5
    # other options
    cfg.fedproto.show_verbose = False  # Weather display verbose loss information
    # ---------------------------------------------------------------------- #
    # (FedPCL) Federated Learning from Pre-Trained Models: A Contrastive Learning Approach
    # ---------------------------------------------------------------------- #
    cfg.fedpcl = CN()
    # for debug
    cfg.model.fedpcl = CN()
    cfg.model.fedpcl.model_weight_dir = './contrib/model_weight'
    cfg.model.fedpcl.input_size = 512
    cfg.model.fedpcl.output_dim = 256

    cfg.fedpcl.debug = False

    cfg.fedpcl.show_verbose = False  # Weather display verbose loss information

    # ---------------------------------------------------------------------- #
    # (FML) Federated Mutual Learning related options
    # ---------------------------------------------------------------------- #
    cfg.fml = CN()
    cfg.fml.alpha = 0.5
    cfg.fml.beta = 0.5
    cfg.model.T = 5  # 临时变量

    # Model related options
    cfg.fml.meme_model = CN()
    cfg.fml.meme_model.type = 'CNN'
    cfg.fml.meme_model.hidden = 256
    cfg.fml.meme_model.dropout = 0.5
    cfg.fml.meme_model.in_channels = 0
    cfg.fml.meme_model.out_channels = 1
    cfg.fml.meme_model.layer = 2
    cfg.fml.meme_model.T = 5  # TODO: 临时变量
    cfg.fml.meme_model.task = "CV_low"
    cfg.fml.meme_model.warpFC = False

    # ---------------------------------------------------------------------- #
    # FedHeNN related options
    # ---------------------------------------------------------------------- #
    cfg.fedhenn = CN()
    cfg.fedhenn.eta = 0.001  # weight of proto loss

    # ---------------------------------------------------------------------- #
    # (FSFL) Few-Shot Model Agnostic Federated Learning
    # ---------------------------------------------------------------------- #
    cfg.fsfl = CN()

    # dataset realated option to verify the correctness of the reproduction
    cfg.fsfl.public_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 参考源代码
    cfg.fsfl.private_classes = [10, 11, 12, 13, 14, 15]  # 参考源代码
    cfg.fsfl.N_samples_per_class = 12

    # Latent Embedding Adaptation
    # Step1: domain identifier realated option
    cfg.fsfl.domain_identifier_epochs = 4
    cfg.fsfl.domain_identifier_batch_size = 30
    cfg.fsfl.DI_optimizer = CN()
    cfg.fsfl.DI_optimizer.type = 'Adam'
    cfg.fsfl.DI_optimizer.lr = 0.001  # 参考源代码
    cfg.fsfl.DI_optimizer.weight_decay = 1e-4  # 参考源代码

    # Step2: local gan training related option
    cfg.fsfl.gan_local_epochs = 4  # 参考源代码
    cfg.fsfl.DI_optimizer_step_2 = CN()
    cfg.fsfl.DI_optimizer_step_2.type = 'Adam'
    cfg.fsfl.DI_optimizer_step_2.lr = 0.0001  # 参考源代码
    cfg.fsfl.DI_optimizer_step_2.weight_decay = 1e-4  # 参考源代码

    # model agnostic federated learning related option
    cfg.fsfl.collaborative_epoch = 1  # 参考源代码
    cfg.fsfl.collaborative_num_samples_epochs = 5000
    cfg.fsfl.MAFL_batch_size = 256  # 参考源代码

    # model related options
    cfg.model.fsfl_cnn_layer1_out_channels = 128
    cfg.model.fsfl_cnn_layer2_out_channels = 512

    # ---------------------------------------------------------------------- #
    # FCCL related options
    # ---------------------------------------------------------------------- #
    cfg.fccl = CN()
    cfg.fccl.structure = 'homogeneity'
    cfg.fccl.beta = 0.1
    cfg.fccl.off_diag_weight = 0.0051
    cfg.fccl.loss_dual_weight = 1
    cfg.fccl.pub_aug = 'weak'

    # ---------------------------------------------------------------------- #
    # DENSE: Data-Free One-Shot Federated Learning
    # ---------------------------------------------------------------------- #
    cfg.DENSE = CN()
    cfg.DENSE.pretrain_epoch = 300
    cfg.DENSE.model_heterogeneous = True
    cfg.DENSE.nz = 256  # number of total iterations in each epoch
    cfg.DENSE.g_steps = 256  # number of iterations for generation
    cfg.DENSE.lr_g = 1e-3  # initial learning rate for generation
    cfg.DENSE.synthesis_batch_size = 256
    cfg.DENSE.sample_batch_size = 256
    cfg.DENSE.adv = 0  # scaling factor for adv loss
    cfg.DENSE.bn = 0  # scaling factor for BN regularization
    cfg.DENSE.oh = 0  # scaling factor for one hot loss (cross entropy)
    cfg.DENSE.act = 0  # scaling factor for activation loss used in DAFL
    cfg.DENSE.save_dir = './contrib/synthesis'
    cfg.DENSE.T = 1.0

    # ---------------------------------------------------------------------- #
    # (FedGH) FedGH: Heterogeneous Federated Learning with Generalized Global Header
    # ---------------------------------------------------------------------- #
    cfg.FedGH = CN()
    cfg.FedGH.server_optimizer = CN()
    cfg.FedGH.server_optimizer.type = 'Adam'
    cfg.FedGH.server_optimizer.lr = 0.001
    cfg.FedGH.server_optimizer.weight_decay = 0.
    cfg.FedGH.server_optimizer.momentum = 0.9

    # ---------------------------------------------------------------------- #
    # (FedDistill) Communication-Efficient On-Device Machine Learning:
    # Federated Distillation and Augmentation under Non-IID Private Data
    # ---------------------------------------------------------------------- #
    cfg.FedDistill = CN()
    cfg.FedDistill.gamma = 1.0

    # global_logit_type=0 or 1.
    # 如果是0，则返回的某个类的全局logits为这个类的所有上传的本地logits的平均；
    # 如果是1，则为每一个client生成独特的全局logits，计算方式为排除掉当前client上传的logits求和，然后初一client_num-1。
    # 1这种方案参考于原文的Algorithm 1
    cfg.FedDistill.global_logit_type = 0

    # ---------------------------------------------------------------------- #
    # (FedKD) Communication-efficient federated learning via knowledge distillation
    # ---------------------------------------------------------------------- #
    cfg.fedkd = CN()

    cfg.fedkd.tmin = 0.95  # refer to https://github.com/wuch15/FedKD/blob/main/parameters.py#L46-L47
    cfg.fedkd.tmax = 0.98

    cfg.fedkd.use_SVD = True

    # ------------------------------------------------------------------------ #
    # (FPL) Rethinking Federated Learning with Domain Shift: A Prototype View
    # ------------------------------------------------------------------------ #
    cfg.fpl = CN()
    cfg.fpl.temperature = 0.02
    cfg.fpl.share_model_weight = False

    # ------------------------------------------------------------------------ #
    # FedAPEN: Personalized Cross-silo Federated Learning with Adaptability to Statistical Heterogeneity
    # ------------------------------------------------------------------------ #
    cfg.fedapen = CN()
    cfg.fedapen.adaptability_ratio = 0.05
    cfg.fedapen.epoch_for_learn_weight = 1
    # ------------------------------------------------------------------------ #
    # POI
    # ------------------------------------------------------------------------ #
    cfg.MHFL.tau = 1.0
    cfg.poi = CN()
    cfg.poi.use_knn = False
    cfg.poi.temp_idx = 1
    cfg.poi.LP_layer = 1
    cfg.poi.LP_alpha = 0.5
    cfg.poi.tau1 = 10
    cfg.poi.proto_agg_type = 'train_loss'
    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_mhfl_cfg)

    # ----------------------------------wait to delete----------------------- #
    # pass


def assert_mhfl_cfg(cfg):
    num_classes_dict = {
        "cora": 7,
        "citeseer": 6,
        'pubmed': 3,
        "CIFAR10@torchvision": 10,
        "office_caltech": 10,
        "SVHN@torchvision": 10,
        'dblp_conf': 4,
        'ogbn_arxiv': 40,
        'computers': 10,
        'photo': 8,
        'arxiv': 40,
    }

    if cfg.model.num_classes == -1 and cfg.data.type in num_classes_dict:
        cfg.model.num_classes = num_classes_dict[cfg.data.type]
        logger.warning(f"Detected that cfg.model.num_classes is not set and the dataset is {cfg.data.type}."
                       f"We specify cfg.model.num_class as {cfg.model.num_classes}")
    elif cfg.model.num_classes == -1:
        cfg.model.num_classes = 10
        logger.warning(
            f"Detected that cfg.model.num_classes is not set and the nunber of classes for {cfg.data.type} is not predefined."
            f"We specify cfg.model.num_classes as {cfg.model.num_classes}")

    if cfg.model.warpFC and cfg.model.feature_dim == -1:
        raise ValueError(f"When cfg.model.warpFC is True, the value of cfg.model.feature_dim must be specified.")
    elif cfg.model.warpFC and cfg.model.out_channels != cfg.model.feature_dim:
        cfg.model.out_channels = cfg.model.feature_dim
        logger.info(
            f"We specify the original model's out_channels as cfg.model.feature_dim {cfg.model.feature_dim} when cfg.model.warpFC is True")

    emb_file_path = cfg.MHFL.emb_file_path
    if cfg.vis_embedding and emb_file_path == None:
        raise ValueError(f"Detected that cfg.vis_embedding is True. Please specify the value of cfg.emb_file_path.")
    elif cfg.vis_embedding and emb_file_path is not None and len(os.listdir(emb_file_path)) != 0:
        from federatedscope.contrib.common_utils import delete_embeeding_files
        delete_embeeding_files(emb_file_path)
        os.mkdir(emb_file_path)


register_config("model_heterogeneity", extend_model_heterogeneous_cfg)
