from federatedscope.core.configs.config import CN
from federatedscope.core.configs.yacs_config import Argument
from federatedscope.register import register_config
import logging

logger = logging.getLogger(__name__)
def extend_graph_model_heterogeneous_cfg(cfg):
    cfg.model.LP_alpha=0.9 # for label propagation algorithm

register_config("graph_model_heterogeneity", extend_graph_model_heterogeneous_cfg)