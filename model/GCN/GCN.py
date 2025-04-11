import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import dgl
import dgl.nn.pytorch as dglnn

###############################################################################
# 1) 改写的 GraphConvLayer：用 HeteroGraphConv 适配多 etype
###############################################################################
class GraphConvLayer(nn.Module):
    """
    - 与原先的单一 GraphConv 不同，这里使用 HeteroGraphConv 来兼容多边类型异构图。
    - 不改变外部的调用方式 (layer(block, node_features))。
    """
    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        in_feat: int,
        out_feat: int,
        dropout=0.0,
        activation=None,
        self_loop=False
    ):
        super(GraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.activation = activation
        self.self_loop = self_loop

        # 创建一个 HeteroGraphConv，内部对每个 etype 都使用同一个(或不同)的 GraphConv
        # 这里为了简洁，给每个 etype 都建一个相同配置的 GraphConv
        # 如果想与原先 RGCN 区分 etype，你也可在此区分不同的 weight/bias
        mod_dict = {}
        for etype in graph.etypes:
            mod_dict[etype] = dglnn.GraphConv(
                in_feat, out_feat, norm='right', weight=True, bias=True
            )

        # aggregate='sum' 表示对不同 etype 的卷积结果做求和，也可以换成 'mean'、'stack' 等
        self.hetero_conv = dglnn.HeteroGraphConv(mod_dict, aggregate='sum')

        # 如果需要显式可训练的 self-loop 权重
        if self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        else:
            self.loop_weight = None

        self.dropout = nn.Dropout(dropout)

    def forward(self, block: dgl.DGLHeteroGraph, node_features: dict):
        """
        与原接口保持一致：输入 block + dict[ntype->feat]，输出 dict[ntype->updated_feat]
        """
        # HeteroGraphConv 接口要求传入 (input_src, input_dst) 或一个 dict
        # 若是普通全连接，且 block 的 src/dst 特征都一样，直接用 node_features 即可：
        #   output_dict = self.hetero_conv(block, node_features)
        # 如果是 bipartite，需要传 (dict_src, dict_dst)。此处只演示一般情况。
        output_dict = self.hetero_conv(block, node_features)

        # 对每个节点类型做后处理(加 self-loop、激活函数、dropout)
        new_node_features = {}
        for ntype in output_dict:
            out = output_dict[ntype]
            # self-loop
            if self.loop_weight is not None:
                # 只有在 out 的 batch_size 与 node_features[ntype] 相同，才做 self-loop
                if out.shape[0] == node_features[ntype].shape[0] and out.shape[0] > 0:
                    out = out + node_features[ntype] @ self.loop_weight

            # 如果有自定义 bias，也可在此加
            # out = out + self.h_bias  (若需要手动 bias)

            # activation
            if self.activation:
                out = self.activation(out)

            out = self.dropout(out)
            new_node_features[ntype] = out

        return new_node_features

###############################################################################
# 2) 改写的 GCN 模型：仅替换 Layer，其他逻辑不变
###############################################################################
class GCN(nn.Module):
    """
    与之前的 RGCN 类似的结构，但去掉了 basis、relation 等概念。
    只是在每层使用上面改造后的 GraphConvLayer(HeteroGraphConv) 来支持多 etype。
    外部接口保持不变：projection_layer、forward(blocks, node_features)、inference(...)。
    """
    def __init__(
        self,
        graph: dgl.DGLHeteroGraph,
        input_dim_dict: dict,
        hidden_sizes: list,
        dropout=0.0,
        use_self_loop=False
    ):
        super(GCN, self).__init__()
        self.input_dim_dict = input_dim_dict
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        # 不同类型节点的特征，映射到同一维度
        self.projection_layer = nn.ModuleDict({
            ntype: nn.Linear(input_dim_dict[ntype], hidden_sizes[0])
            for ntype in input_dim_dict
        })

        # 建立多层
        self.layers = nn.ModuleList()
        # 第一层
        self.layers.append(
            GraphConvLayer(
                graph,
                in_feat=self.hidden_sizes[0],
                out_feat=self.hidden_sizes[0],
                dropout=self.dropout,
                activation=F.relu,
                self_loop=self.use_self_loop
            )
        )
        # 后续层
        for i in range(1, len(self.hidden_sizes)):
            self.layers.append(
                GraphConvLayer(
                    graph,
                    in_feat=self.hidden_sizes[i - 1],
                    out_feat=self.hidden_sizes[i],
                    dropout=self.dropout,
                    activation=F.relu,
                    self_loop=self.use_self_loop
                )
            )

    def forward(self, blocks: list, node_features: dict):
        """
        与原 RGCN forward 保持一致：
          - blocks: 每层采样得到的异构子图列表
          - node_features: {ntype: tensor}, 各类型节点特征
        """
        # 先做 projection
        for ntype in node_features:
            node_features[ntype] = self.projection_layer[ntype](node_features[ntype])

        # 多层堆叠
        for block, layer in zip(blocks, self.layers):
            node_features = layer(block, node_features)

        return node_features

    def inference(self, graph: dgl.DGLHeteroGraph, node_features: dict, device: str):
        """
        全图推理，与原接口一致。
        """
        with torch.no_grad():
            for layer_idx, layer in enumerate(self.layers):
                # 创建容器存放整图节点输出
                y = {
                    ntype: torch.zeros(
                        graph.number_of_nodes(ntype), self.hidden_sizes[layer_idx]
                    )
                    for ntype in graph.ntypes
                }

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.DataLoader(
                    graph,
                    {
                        ntype: torch.arange(graph.number_of_nodes(ntype), dtype=torch.int32).to(device)
                        for ntype in graph.ntypes
                    },
                    sampler,
                    batch_size=1024,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0
                )

                tqdm_dataloader = tqdm(dataloader, ncols=120)
                for batch, (input_nodes, output_nodes, blocks) in enumerate(tqdm_dataloader):
                    block = blocks[0].to(device)

                    # 取输入特征
                    input_feats = {}
                    for ntype in input_nodes:
                        input_feats[ntype] = node_features[ntype][input_nodes[ntype].cpu()].to(device)

                    # 第 0 层时还需做 projection
                    if layer_idx == 0:
                        for ntype in input_feats:
                            input_feats[ntype] = self.projection_layer[ntype](input_feats[ntype])

                    # 做卷积
                    h = layer(block, input_feats)

                    # 保存结果到 y
                    for ntype in h.keys():
                        if ntype in output_nodes:  # 这个判断一般是 True
                            y[ntype][output_nodes[ntype]] = h[ntype].cpu()

                    tqdm_dataloader.set_description(
                        f"Inference batch={batch}, layer={layer_idx}"
                    )

                # 将整图特征更新供下一层使用
                node_features = y

        return node_features
