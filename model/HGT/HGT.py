import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from tqdm import tqdm


class HGTLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,          # dict[str -> int]，节点类型到编号
        edge_dict,          # dict[str -> int]，边类型到编号
        n_heads,
        dropout=0.2,
        use_norm=False,
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm

        # 针对每种节点类型的线性映射
        self.k_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_types):
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        # 关系参数
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))

        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        # 参数初始化
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, block, h_dict):
        """
        block: dgl.Block，小批量采样后本层的子图
        h_dict: dict[str -> Tensor]，每种节点类型的特征
            - 一般形状: [block.num_src_nodes(ntype), in_dim] 或 [block.num_dst_nodes(ntype), in_dim]
        """
        with block.local_scope():
            # 1) 针对每个 (srctype, etype, dsttype) 做子边类型级别的处理
            for srctype, etype, dsttype in block.canonical_etypes:
                sub_block = block[srctype, etype, dsttype]

                # 拿到该子边类型的源、目标节点在本层 block 的局部 ID
                sub_src_ids = sub_block.srcnodes(srctype)
                sub_dst_ids = sub_block.dstnodes(dsttype)

                # 从 h_dict[...] 中索引出对应子集
                k_src = h_dict[srctype][sub_src_ids]
                v_src = h_dict[srctype][sub_src_ids]
                q_dst = h_dict[dsttype][sub_dst_ids]

                # 分别调用该节点类型对应的线性层
                k_linear = self.k_linears[self.node_dict[srctype]]
                v_linear = self.v_linears[self.node_dict[srctype]]
                q_linear = self.q_linears[self.node_dict[dsttype]]

                k = k_linear(k_src).view(-1, self.n_heads, self.d_k)
                v = v_linear(v_src).view(-1, self.n_heads, self.d_k)
                q = q_linear(q_dst).view(-1, self.n_heads, self.d_k)

                # 根据 etype 找到对应关系参数
                e_id = self.edge_dict[etype]
                relation_att = self.relation_att[e_id]  # [n_heads, d_k, d_k]
                relation_pri = self.relation_pri[e_id]  # [n_heads]
                relation_msg = self.relation_msg[e_id]  # [n_heads, d_k, d_k]

                # 乘上关系矩阵
                k = torch.einsum("bij,ijk->bik", k, relation_att)  # [#src, n_heads, d_k]
                v = torch.einsum("bij,ijk->bik", v, relation_msg)

                # 存入 sub_block 的 src/dstdata
                sub_block.srcdata["k"] = k
                sub_block.srcdata[f"v_{e_id}"] = v
                sub_block.dstdata["q"] = q

                # ★ 修改点：计算注意力分数时只在 d_k 维度求和，不应将多头维度一起求和
                sub_block.apply_edges(lambda edges: {'t': (edges.dst['q'] * edges.src['k']).sum(dim=-1)})
                attn_score = sub_block.edata.pop("t")  # shape: [num_edges, n_heads]
                attn_score = attn_score * relation_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_block, attn_score, norm_by="dst")
                sub_block.edata["t"] = attn_score.unsqueeze(-1)  # shape: [num_edges, n_heads, 1]

            # 2) multi_update_all 做消息聚合
            update_dict = {}
            for srctype, etype, dsttype in block.canonical_etypes:
                e_id = self.edge_dict[etype]
                update_dict[(srctype, etype, dsttype)] = (
                    fn.u_mul_e(f"v_{e_id}", "t", "m"),   # message_func
                    fn.sum("m", "t"),                     # reduce_func
                )
            block.multi_update_all(update_dict, cross_reducer="mean")

            # 3) 目标节点残差 + 非线性
            new_h = {}
            for ntype in block.ntypes:
                n_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])

                # 若本层 block 中该 ntype 没有入边，则 "t" 不存在，需要做 0 填充
                if "t" in block.nodes[ntype].data:
                    t = block.nodes[ntype].data["t"].view(-1, self.out_dim)
                else:
                    t = torch.zeros(
                        (block.num_dst_nodes(ntype), self.out_dim),
                        device=h_dict[ntype].device,
                    )

                # 先做线性映射和 dropout
                out = self.drop(self.a_linears[n_id](t))

                # ★ 核心：这里要与 h_dict[ntype] 的【目标节点】对齐
                dst_ids = block.dstnodes(ntype)
                h_dst = h_dict[ntype][dst_ids]  # [num_dst_nodes(ntype), out_dim]

                # 残差连接
                out = out * alpha + h_dst * (1 - alpha)

                if self.use_norm:
                    out = self.norms[n_id](out)

                new_h[ntype] = out

            return new_h


class HGT(nn.Module):
    def __init__(
        self,
        G,
        node_dict,          # dict[str -> int]
        edge_dict,          # dict[str -> int]
        n_hid,
        n_out,
        n_layers,
        n_heads,
        use_norm=True,
    ):
        super(HGT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers

        # adapt_ws：将每种类型节点的输入特征投影到 n_hid 维度
        self.adapt_ws = nn.ModuleList()
        for ntype in G.ntypes:
            in_dim = G.nodes[ntype].data["feat"].shape[-1]
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))

        # 叠 n_layers 层 HGTLayer
        self.gcs = nn.ModuleList()
        for _ in range(n_layers):
            self.gcs.append(
                HGTLayer(
                    n_hid,
                    n_hid,
                    node_dict,
                    edge_dict,
                    n_heads,
                    use_norm=use_norm,
                )
            )

        # 为每种节点类型分配独立的输出层
        self.outs = nn.ModuleList()
        for _ in G.ntypes:
            self.outs.append(nn.Linear(n_hid, n_out))

    def forward(self, blocks, node_features):
        """
        blocks: List[dgl.Block]，长度 = self.n_layers
        node_features: dict[str -> Tensor]，传入该 batch 里每种节点类型的特征
        """
        # 1) 特征投影
        for ntype in node_features:
            n_id = self.node_dict[ntype]
            node_features[ntype] = F.gelu(self.adapt_ws[n_id](node_features[ntype]))

        # 2) 逐层调用 HGTLayer
        h = node_features
        for i in range(self.n_layers):
            h = self.gcs[i](blocks[i], h)

        # 3) 各节点类型的输出层
        out_dict = {}
        for ntype in h:
            n_id = self.node_dict[ntype]
            out_dict[ntype] = self.outs[n_id](h[ntype])
        return out_dict

    def inference(self, graph, device):
        self.eval()
        with torch.no_grad():
            # 1. 初始化节点特征
            h = {}
            for ntype in graph.ntypes:
                n_id = self.node_dict[ntype]
                h[ntype] = F.gelu(
                    self.adapt_ws[n_id](graph.nodes[ntype].data['feat']).to(device)
                )

            # 2. 分层推断
            for i, layer in enumerate(self.gcs):
                y = {ntype: torch.zeros(graph.num_nodes(ntype), self.n_hid, device='cpu')
                     for ntype in graph.ntypes}

                sampler = MultiLayerFullNeighborSampler(1)
                dataloader = DataLoader(
                    graph,
                    {ntype: torch.arange(graph.num_nodes(ntype), dtype=torch.int32).to(device)
                     for ntype in graph.ntypes},
                    sampler,
                    batch_size=1024,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0
                )

                tqdm_dataloader = tqdm(dataloader, desc=f'Inference Layer {i+1}/{self.n_layers}', ncols=120)
                for input_nodes, output_nodes, blocks in tqdm_dataloader:
                    blocks = [b.to(device) for b in blocks]

                    input_features = {}
                    for ntype in input_nodes:
                        input_features[ntype] = h[ntype][input_nodes[ntype].cpu()].to(device)

                    h_new = layer(blocks[0], input_features)

                    for ntype in h_new:
                        dst_nids = blocks[0].dstdata[dgl.NID][ntype]
                        y[ntype][dst_nids] = h_new[ntype].cpu()

                h = y

            # 3. 最终输出层
            final_repr = {}
            for ntype in h:
                n_id = self.node_dict[ntype]
                final_repr[ntype] = self.outs[n_id](h[ntype].to(device))
            return final_repr
