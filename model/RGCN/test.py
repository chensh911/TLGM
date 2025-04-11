import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn

###############################################################################
# 统一的 Prompt 层
###############################################################################
class UnifiedPromptLayer(nn.Module):
    """
    一个统一的 prompt 层，可支持多种 prompt 策略：
      - 'weighted_sum'：对输入特征做元素级加权（乘上一个可学习参数）并激活；
      - 'cat'：将一个可学习的 prompt 向量与输入特征拼接；
      - 'sum'：将输入特征与一个可学习的 prompt 向量相加。
    
    同时可选地支持异构 prompt（即在消息传递时额外加入边上的提示信息）。
    
    参数：
      in_feat: 输入特征维度
      prompt_dim: 当 mode 为 'cat' 或 'sum' 时，提示向量的维度（对于 weighted_sum 模式可忽略）
      mode: 使用的 prompt 模式，取值 'weighted_sum'、'cat' 或 'sum'
      hetero: 是否启用异构提示（默认为 False）
      hetero_dim: 如果启用异构提示，则必须提供边提示的维度
      dropout: dropout 率
    """
    def __init__(self, in_feat, prompt_dim=None, mode='weighted_sum', hetero=False, hetero_dim=None, dropout=0.0):
        super(UnifiedPromptLayer, self).__init__()
        self.mode = mode
        self.hetero = hetero
        if mode == 'weighted_sum':
            # weighted_sum 模式直接使用一个形状为 (1, in_feat) 的参数
            self.weight = nn.Parameter(torch.Tensor(1, in_feat))
        elif mode in ['cat', 'sum']:
            assert prompt_dim is not None, "mode为'cat'或'sum'时，必须指定 prompt_dim"
            self.prompt = nn.Parameter(torch.Tensor(1, prompt_dim))
        else:
            raise ValueError("不支持的 mode: {}".format(mode))
        
        if self.hetero:
            assert hetero_dim is not None, "启用 hetero 时必须指定 hetero_dim"
            self.hetero_prompt = nn.Parameter(torch.Tensor(1, hetero_dim))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.mode == 'weighted_sum':
            nn.init.xavier_uniform_(self.weight)
        elif self.mode in ['cat', 'sum']:
            nn.init.xavier_uniform_(self.prompt)
        if self.hetero:
            nn.init.xavier_uniform_(self.hetero_prompt)
    
    def forward(self, graph, node_features, edge_features=None):
        """
        对图中每个节点的特征应用 prompt 处理。
        如果启用了 hetero 并且提供了 edge_features，则在消息传递时将边上的提示信息加入。
        """
        if self.mode == 'weighted_sum':
            # 逐元素加权后使用 ELU 激活
            h = F.elu(node_features * self.weight)
        elif self.mode == 'cat':
            prompt_vec = self.prompt.expand(node_features.size(0), -1)
            h = torch.cat([node_features, prompt_vec], dim=1)
        elif self.mode == 'sum':
            prompt_vec = self.prompt.expand(node_features.size(0), node_features.size(1))
            h = node_features + prompt_vec
        else:
            raise ValueError("不支持的 mode")
        
        if self.hetero and edge_features is not None:
            num_edges = graph.number_of_edges()
            hetero_vec = self.hetero_prompt.expand(num_edges, -1)
            # 将边特征（若有）和异构提示拼接到消息中
            graph.edata['ef'] = edge_features if edge_features is not None else torch.empty(num_edges, 0).to(h.device)
            graph.edata['hp'] = hetero_vec
            graph.srcdata['h'] = h
            def message_func(edges):
                return {'m': torch.cat([edges.src['h'], edges.data['ef'], edges.data['hp']], dim=1)}
            graph.update_all(message_func, fn.sum('m', 'h_out'))
            out = graph.dstdata['h_out']
        else:
            graph.srcdata['h'] = h
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_out'))
            out = graph.dstdata['h_out']
        return self.dropout(out)

###############################################################################
# 关系图卷积层（支持异构图）
###############################################################################
class RelGraphConvLayer(nn.Module):
    """
    一个关系图卷积层，适用于异构图。采用 dgl 的 HeteroGraphConv 对每种边类型单独计算图卷积，
    并可选择使用 basis 分解来减少参数量。
    """
    def __init__(self, graph: dgl.DGLHeteroGraph, in_feat, out_feat, num_bases, dropout=0.0,
                 activation=None, self_loop=False):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_bases = num_bases
        self.etypes = graph.etypes
        self.activation = activation
        self.self_loop = self_loop
        
        # 对每个边类型使用 GraphConv（不包含可训练参数，后面由 weight/basis 控制）
        self.relation_conv = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for etype in self.etypes
        })
        self.use_basis = 0 < num_bases < len(self.etypes)
        if self.use_basis:
            self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.etypes))
        else:
            self.weight = nn.Parameter(torch.randn(len(self.etypes), in_feat, out_feat))
        
        self.h_bias = nn.Parameter(torch.zeros(out_feat))
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.randn(in_feat, out_feat))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if not self.use_basis:
            nn.init.xavier_uniform_(self.weight, gain=gain)
        nn.init.zeros_(self.h_bias)
        if self.self_loop:
            nn.init.xavier_uniform_(self.loop_weight, gain=gain)
    
    def forward(self, graph: dgl.DGLHeteroGraph, node_features: dict):
        graph = graph.local_var()
        input_src = node_features
        if graph.is_block:
            input_dst = {ntype: node_features[ntype][:graph.number_of_dst_nodes(ntype)]
                         for ntype in node_features}
        else:
            input_dst = node_features
        
        # 根据是否使用 basis 选择权重
        weight = self.basis() if self.use_basis else self.weight
        weight_dict = {self.etypes[i]: {'weight': w.squeeze(0)}
                       for i, w in enumerate(torch.split(weight, 1, dim=0))}
        
        output_features = self.relation_conv(graph, (input_src, input_dst), mod_kwargs=weight_dict)
        dst_node_features = {}
        for ntype in output_features:
            h = output_features[ntype]
            if self.self_loop:
                h = h + torch.matmul(input_dst[ntype], self.loop_weight)
            h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            dst_node_features[ntype] = self.dropout(h)
        return dst_node_features

###############################################################################
# 统一的 GNN 模型
###############################################################################
class UnifiedGNN(nn.Module):
    """
    一个统一的 GNN 模型，适用于异构图，并支持在最后接入可配置的 prompt 层。
    
    参数：
      graph: DGLHeteroGraph 对象
      input_dim_dict: 字典，映射每种节点类型到其输入特征维度
      hidden_sizes: 隐藏层的尺寸列表
      num_bases: 关系卷积中使用的 basis 个数
      dropout: dropout 率
      use_self_loop: 是否在卷积中使用自环
      prompt_config: prompt 层的配置字典（若为 None 则不使用 prompt 层）
                     示例：
                     {
                        'in_feat': hidden_sizes[-1],
                        'prompt_dim': hidden_sizes[-1],  # 或其它尺寸
                        'mode': 'weighted_sum',   # 可选 'weighted_sum', 'cat', 'sum'
                        'hetero': False,          # 是否使用异构 prompt
                        'hetero_dim': None,       # 若 hetero=True，指定维度
                        'dropout': dropout
                     }
    """
    def __init__(self, graph: dgl.DGLHeteroGraph, input_dim_dict: dict, hidden_sizes: list,
                 num_bases: int, dropout=0.0, use_self_loop=False, prompt_config=None):
        super(UnifiedGNN, self).__init__()
        self.graph = graph
        self.input_dim_dict = input_dim_dict
        self.hidden_sizes = hidden_sizes
        self.num_bases = num_bases
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        
        # 每个节点类型通过投影层对齐到统一隐藏空间
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_sizes[0])
            for ntype in input_dim_dict
        })
        
        self.layers = nn.ModuleList()
        self.layers.append(
            RelGraphConvLayer(graph, hidden_sizes[0], hidden_sizes[0],
                              num_bases, dropout=dropout, activation=F.relu,
                              self_loop=use_self_loop)
        )
        for i in range(1, len(hidden_sizes)):
            self.layers.append(
                RelGraphConvLayer(graph, hidden_sizes[i-1], hidden_sizes[i],
                                  num_bases, dropout=dropout, activation=F.relu,
                                  self_loop=use_self_loop)
            )
        
        # 如果提供了 prompt 配置，则接入统一的 prompt 层
        if prompt_config is not None:
            self.prompt_layer = UnifiedPromptLayer(**prompt_config)
        else:
            self.prompt_layer = None
    
    def forward(self, blocks: list, node_features: dict):
        # 对各类型节点先进行投影
        for ntype in node_features:
            node_features[ntype] = self.projection_layer[ntype](node_features[ntype])
        # 逐层进行关系图卷积
        for block, layer in zip(blocks, self.layers):
            node_features = layer(block, node_features)
        # 若设置了 prompt 层，则对每个节点类型的特征加以融合
        if self.prompt_layer is not None:
            for ntype in node_features:
                node_features[ntype] = self.prompt_layer(self.graph, node_features[ntype])
        return node_features
