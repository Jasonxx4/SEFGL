# """
#     refer to: xtract the Knowledge of Graph Neural Networks and Go Beyond it: An Effective Knowledge Distillation Framework
# """
# from federatedscope.register import register_model
# from typing import Callable, Optional
# import torch
# import torch.nn as nn
# from torch import Tensor
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# from torch_geometric.typing import Adj, OptTensor, SparseTensor
# from torch_geometric.utils import one_hot, spmm
# from torch_geometric.data import Data
# from torch_geometric.utils import scatter
#
#
# class PLPConv_PYG(MessagePassing):
#     def __init__(self, in_channels,
#                  out_channels,
#                  num_class,
#                  node_num,
#                  feat_drop,
#                  attn_drop,
#                  residual=False,
#                  activation=None,
#                  mlp_layers=0,
#                  ):
#         super().__init__(aggr='add')
#         self._in_src_feats = in_channels
#         self._out_feats = out_channels
#         self.lr_alpha = nn.Parameter(torch.zeros(size=(node_num, 1)))
#         self.fc_emb = nn.Parameter(torch.FloatTensor(size=(node_num, 1)))  # 个人理解：对应CPF公式(6)
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.mlp_layers = mlp_layers
#         if self.mlp_layers > 0:
#             self.mlp = MLP2(self.mlp_layers, self._in_src_feats, out_feats, num_class, feat_drop)
#         if residual:
#             if self._in_dst_feats != out_channels:
#                 self.res_fc = nn.Linear(
#                     self._in_dst_feats, out_channels, bias=False)
#             else:
#                 self.res_fc = torch.nn.Identity()
#         else:
#             self.register_buffer('res_fc', None)
#         self.reset_parameters()
#         self.activation = activation
#
#     def forward(self, data, feat, soft_label):
#         if isinstance(data, Data):
#             x, y, edge_index = data.x, data.y, data.edge_index
#         else:
#             raise TypeError('Unsupported data type!')
#         row, col = edge_index
#
#         feat_src = self.feat_drop(self.fc_emb)
#         feat_dst = h_dst = torch.zeros(data.num_nodes(), device=data.device)
#         el = feat_src
#         er = feat_dst
#         cog_label = soft_label
#
#         node_attr = el[row] + er[col]
#
#         # message passing
#         out = self.propagate(edge_index, x=cog_label,node_attr=node_attr,
#                              size=None)
#
#     def message(self, x,edge_index,edge_attr):
#         pass
#
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.fc_emb, gain=gain)
#
#
# class PLPConv(nn.Module):
#     def __init__(self,
#                  in_feats,
#                  out_feats,
#                  num_class,
#                  node_num,
#                  feat_drop=0.,
#                  attn_drop=0.,
#                  residual=False,
#                  activation=None,
#                  mlp_layers=0,
#                  allow_zero_in_degree=False,
#                  ptype='ind'):
#         super(PLPConv, self).__init__()
#         self._in_src_feats = in_feats
#         self._out_feats = out_feats
#         self.lr_alpha = nn.Parameter(torch.zeros(size=(node_num, 1)))
#         self.fc_emb = nn.Parameter(torch.FloatTensor(size=(node_num, 1)))  # 个人理解：对应CPF公式(6)
#         self.feat_drop = nn.Dropout(feat_drop)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.mlp_layers = mlp_layers
#         if self.mlp_layers > 0:
#             self.mlp = MLP2(self.mlp_layers, self._in_src_feats, out_feats, num_class, feat_drop)
#         if residual:
#             if self._in_dst_feats != out_feats:
#                 self.res_fc = nn.Linear(
#                     self._in_dst_feats, out_feats, bias=False)
#             else:
#                 self.res_fc = torch.nn.Identity()
#         else:
#             self.register_buffer('res_fc', None)
#         self.reset_parameters()
#         self.activation = activation
#
#     def reset_parameters(self):
#         """
#
#         Description
#         -----------
#         Reinitialize learnable parameters.
#         Note
#         ----
#         The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
#         The attention weights are using xavier initialization method.
#         """
#         gain = nn.init.calculate_gain('relu')
#         if isinstance(self.res_fc, nn.Linear):
#             nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.fc_emb, gain=gain)
#
#     def forward(self, graph, feat, soft_label):
#         with graph.local_scope():
#             if not self._allow_zero_in_degree:
#                 if (graph.in_degrees() == 0).any():
#                     raise DGLError('There are 0-in-degree nodes in the graph, '
#                                    'output for those nodes will be invalid. '
#                                    'This is harmful for some applications, '
#                                    'causing silent performance regression. '
#                                    'Adding self-loop on the input graph by '
#                                    'calling `g = dgl.add_self_loop(g)` will resolve '
#                                    'the issue. Setting ``allow_zero_in_degree`` '
#                                    'to be `True` when constructing this module will '
#                                    'suppress the check and let the code run.')
#             if self.ptype == 'ind':
#                 feat_src = h_dst = self.feat_drop(feat)
#                 el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
#                 er = th.zeros(graph.num_nodes(), device=graph.device)
#             elif self.ptype == 'tra':
#                 feat_src = self.feat_drop(self.fc_emb)
#                 feat_dst = h_dst = th.zeros(graph.num_nodes(), device=graph.device)
#                 el = feat_src
#                 er = feat_dst
#             cog_label = soft_label
#             graph.srcdata.update({'ft': cog_label, 'el': el})
#             graph.dstdata.update({'er': er})
#             # # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
#             graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
#             # graph.edata['e'] = th.ones(graph.num_edges(), device=graph.device)  # non-parameterized PLP
#             e = graph.edata.pop('e')
#             # compute softmax
#             graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
#             att = graph.edata['a'].squeeze()
#             # message passing
#             graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
#                              fn.sum('m', 'ft'))
#             if self.mlp_layers > 0:
#                 rst = th.sigmoid(self.lr_alpha) * graph.dstdata['ft'] + \
#                       th.sigmoid(-self.lr_alpha) * self.mlp(feat)
#             else:
#                 rst = graph.dstdata['ft']
#             # residual
#             if self.res_fc is not None:
#                 resval = self.res_fc(h_dst)
#                 rst = rst + resval
#             # activation
#             if self.activation:
#                 rst = self.activation(rst)
#             return rst, att, th.sigmoid(self.lr_alpha).squeeze(), el.squeeze(), er.squeeze()
#             # return rst, att, self.lr_alpha.squeeze()
#
#
# class LabelPropagation(MessagePassing):
#     r"""The label propagation operator from the `"Learning from Labeled and
#     Unlabeled Data with Label Propagation"
#     <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`_ paper
#
#     .. math::
#         \mathbf{Y}^{\prime} = \alpha \cdot \mathbf{D}^{-1/2} \mathbf{A}
#         \mathbf{D}^{-1/2} \mathbf{Y} + (1 - \alpha) \mathbf{Y},
#
#     where unlabeled data is inferred by labeled data via propagation.
#
#     .. note::
#
#         For an example of using the :class:`LabelPropagation`, see
#         `examples/label_prop.py
#         <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
#         label_prop.py>`_.
#
#     Args:
#         num_layers (int): The number of propagations.
#         alpha (float): The :math:`\alpha` coefficient.
#     """
#
#     def __init__(self, num_layers: int, alpha: float):
#         super().__init__(aggr='add')
#         self.num_layers = num_layers
#         self.alpha = alpha
#
#     @torch.no_grad()
#     def forward(
#             self,
#             y: Tensor,
#             edge_index: Adj,
#             mask: OptTensor = None,
#             edge_weight: OptTensor = None,
#             post_step: Optional[Callable[[Tensor], Tensor]] = None,
#     ) -> Tensor:
#         r"""
#         Args:
#             y (torch.Tensor): The ground-truth label information
#                 :math:`\mathbf{Y}`.
#             edge_index (torch.Tensor or SparseTensor): The edge connectivity.
#             mask (torch.Tensor, optional): A mask or index tensor denoting
#                 which nodes are used for label propagation.
#                 (default: :obj:`None`)
#             edge_weight (torch.Tensor, optional): The edge weights.
#                 (default: :obj:`None`)
#             post_step (callable, optional): A post step function specified
#                 to apply after label propagation. If no post step function
#                 is specified, the output will be clamped between 0 and 1.
#                 (default: :obj:`None`)
#         """
#         if y.dtype == torch.long and y.size(0) == y.numel():
#             y = one_hot(y.view(-1))
#
#         out = y
#         if mask is not None:
#             out = torch.zeros_like(y)
#             out[mask] = y[mask]
#
#         if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
#             edge_index = gcn_norm(edge_index, add_self_loops=False)
#         elif isinstance(edge_index, Tensor) and edge_weight is None:
#             edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
#                                                add_self_loops=False)
#
#         res = (1 - self.alpha) * out
#         for _ in range(self.num_layers):
#             # propagate_type: (y: Tensor, edge_weight: OptTensor)
#             out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
#                                  size=None)
#             out.mul_(self.alpha).add_(res)
#             if post_step is not None:
#                 out = post_step(out)
#             else:
#                 out.clamp_(0., 1.)
#
#         return out
#
#     def forward(self, data):
#         if isinstance(data, Data):
#             y, edge_index = data.y, data.edge_index
#         else:
#             raise TypeError('Unsupported data type!')
#
#         if y.dtype == torch.long and y.size(0) == y.numel():
#             y = one_hot(y.view(-1))
#
#         out = y
#
#         if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
#             edge_index = gcn_norm(edge_index, add_self_loops=False)
#
#         res = (1 - self.alpha) * out
#         for _ in range(self.num_layers):
#             # propagate_type: (y: Tensor, edge_weight: OptTensor)
#             out = self.propagate(edge_index, x=out, edge_weight=None,
#                                  size=None)
#             out.mul_(self.alpha).add_(res)
#
#             out.clamp_(0., 1.)
#
#         return out
#
#     def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
#         return x_j if edge_weight is None else edge_weight.view(-1,
#                                                                 1) * x_j  # zhl:x_j用来索引edge_index中第0行（发送消息的节点的embeeding）
#
#     def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
#         return spmm(adj_t, x, reduce=self.aggr)
#
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}(num_layers={self.num_layers}, '
#                 f'alpha={self.alpha})')
#
#
# class MLP2(nn.Module):
#     def __init__(self, num_layers, input_dim, hidden_dim, output_dim, dropout):
#         super(MLP2, self).__init__()
#         self.linear_or_not = True  # default is linear model
#         self.num_layers = num_layers
#         self.dropout = nn.Dropout(dropout)
#         if num_layers < 1:
#             raise ValueError("number of layers should be positive!")
#         elif num_layers == 1:
#             # Linear model ()
#             self.linear = nn.Linear(input_dim, output_dim)
#         else:
#             # Multi-layer model
#             self.linear_or_not = False
#             self.linears = torch.nn.ModuleList()
#             # self.batch_norms = torch.nn.ModuleList()
#             self.linears.append(nn.Linear(input_dim, hidden_dim))
#             for layer in range(num_layers - 2):
#                 self.linears.append(nn.Linear(hidden_dim, hidden_dim))
#             self.linears.append(nn.Linear(hidden_dim, output_dim))
#             # for layer in range(num_layers - 1):
#             #     self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
#
#     def forward(self, x):
#         if self.linear_or_not:  # If linear model
#             return self.linear(self.dropout(x))
#         else:  # If MLP
#             h = x
#             for layer in range(self.num_layers - 1):
#                 h = F.relu(self.linears[layer](self.dropout(h)))
#             return self.linears[self.num_layers - 1](self.dropout(h))
#
#
# def call_mlp(model_config, local_data):
#     if 'PYG_PLP' == model_config.type:
#         model = LabelPropagation(num_layers=model_config.layer, alpha=model_config.LP_alpha)
#         return model
#
#
# register_model('PYG_LP', call_mlp)
