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


class HANLayer(nn.Module):
    """
    HANLayer: Heterogeneous Graph Attention Layer
    This layer performs multi-head attention over different edge types and aggregates
    messages to update node representations.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        node_dict (dict): Dictionary mapping node types to unique IDs.
        edge_dict (dict): Dictionary mapping edge types to unique IDs.
        n_heads (int): Number of attention heads.
        dropout (float, optional): Dropout rate. Default: 0.2.
        use_norm (bool, optional): Whether to use LayerNorm. Default: False.
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,
        edge_dict,
        n_heads,
        dropout=0.2,
        use_norm=False,
    ):
        super(HANLayer, self).__init__()

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

        # Linear layers for Q, K, V transformations
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

        # Relation-specific parameters
        self.relation_pri = nn.Parameter(torch.ones(self.num_relations, self.n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_relations, n_heads, self.d_k, self.d_k))

        # Skip connection and dropout
        self.skip = nn.Parameter(torch.zeros(self.num_types))
        self.drop = nn.Dropout(dropout)

        # Initialize parameters
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)
        nn.init.normal_(self.relation_pri, mean=1.0, std=0.1)

    def forward(self, block, h_dict):
        with block.local_scope():
            # Step 1: Compute attention scores for each edge type
            for srctype, etype, dsttype in block.canonical_etypes:
                sub_block = block[srctype, etype, dsttype]
                e_id = self.edge_dict[etype]

                # Get source and destination node IDs
                src_ids = sub_block.srcnodes(srctype)
                dst_ids = sub_block.dstnodes(dsttype)

                # Get node features
                k_src = h_dict[srctype][src_ids]
                v_src = h_dict[srctype][src_ids]
                q_dst = h_dict[dsttype][dst_ids]

                # Apply linear transformations
                k_linear = self.k_linears[self.node_dict[srctype]]
                v_linear = self.v_linears[self.node_dict[srctype]]
                q_linear = self.q_linears[self.node_dict[dsttype]]

                # Reshape to multi-head format (B, n_heads, d_k)
                k = k_linear(k_src).view(-1, self.n_heads, self.d_k)
                v = v_linear(v_src).view(-1, self.n_heads, self.d_k)
                q = q_linear(q_dst).view(-1, self.n_heads, self.d_k)

                # Apply relation-specific transformations
                rel_att = self.relation_att[e_id]
                rel_pri = self.relation_pri[e_id]
                rel_msg = self.relation_msg[e_id]

                k = torch.einsum("bij,ijk->bik", k, rel_att)
                v = torch.einsum("bij,ijk->bik", v, rel_msg)

                # Store transformed features
                sub_block.srcdata["k"] = k
                sub_block.srcdata[f"v_{e_id}"] = v
                sub_block.dstdata["q"] = q

                # Compute attention scores
                sub_block.apply_edges(fn.v_dot_u("q", "k", "t"))  # (E, n_heads)
                attn_score = sub_block.edata.pop("t")  # (E, n_heads)
                attn_score = attn_score * rel_pri / self.sqrt_dk
                attn_score = edge_softmax(sub_block, attn_score, norm_by="dst")
                sub_block.edata["a"] = attn_score.unsqueeze(-1)  # (E, n_heads, 1)

            # Step 2: Aggregate messages
            update_dict = {}
            for srctype, etype, dsttype in block.canonical_etypes:
                e_id = self.edge_dict[etype]
                update_dict[(srctype, etype, dsttype)] = (
                    fn.u_mul_e(f"v_{e_id}", "a", "m"),  # Message function
                    fn.sum("m", "t"),                  # Reduce function
                )
            block.multi_update_all(update_dict, cross_reducer="mean")

            # Step 3: Apply residual connections and normalization
            new_h = {}
            for ntype in block.ntypes:
                n_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])

                # Get aggregated features
                if "t" in block.nodes[ntype].data:
                    t = block.nodes[ntype].data["t"].view(-1, self.out_dim)
                else:
                    t = torch.zeros(
                        (block.num_dst_nodes(ntype), self.out_dim),
                        device=h_dict[ntype].device,
                    )

                # Apply final linear transformation and dropout
                out = self.drop(self.a_linears[n_id](t))

                # Apply residual connection
                dst_ids = block.dstnodes(ntype)
                h_dst = h_dict[ntype][dst_ids]
                out = out * alpha + h_dst * (1 - alpha)

                # Apply LayerNorm if enabled
                if self.use_norm:
                    out = self.norms[n_id](out)

                new_h[ntype] = out

            return new_h


class HAN(nn.Module):
    """
    多层 HAN 模型（思路与 HGT 类似），不用元路径做显示聚合，而是直接遍历所有 (srctype, etype, dsttype)。
    """

    def __init__(
        self,
        G,
        node_dict,
        edge_dict,
        n_hid,
        n_out,
        n_layers,
        n_heads,
        use_norm=True,
    ):
        super(HAN, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers

        # 1) 针对每种节点类型的输入特征 -> n_hid
        self.adapt_ws = nn.ModuleList()
        for ntype in G.ntypes:
            in_dim = G.nodes[ntype].data["feat"].shape[-1]
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))

        # 2) 叠加 n_layers 个 HANLayer
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                HANLayer(
                    in_dim=n_hid,
                    out_dim=n_hid,
                    node_dict=node_dict,
                    edge_dict=edge_dict,
                    n_heads=n_heads,
                    dropout=0.2,
                    use_norm=use_norm,
                )
            )

        # 3) 每种节点类型的输出层
        self.outs = nn.ModuleList()
        for _ in G.ntypes:
            self.outs.append(nn.Linear(n_hid, n_out))

    def forward(self, blocks, node_features):
        """
        blocks: List[dgl.Block]，长度 = n_layers
        node_features: dict[str -> Tensor]，形如 { ntype: [batch_size_ntype, in_dim], ... }

        返回: dict[str -> Tensor]，每种目标节点类型的输出 (维度 n_out)
        """
        # 先把输入特征投影到 n_hid
        for ntype in node_features:
            n_id = self.node_dict[ntype]
            x = node_features[ntype]
            node_features[ntype] = F.gelu(self.adapt_ws[n_id](x))

        # 逐层调用
        h = node_features
        for layer in self.layers:
            h = layer(blocks[0], h)
            blocks = blocks[1:]  # 用掉一层 Block

        # 最终输出映射
        out_dict = {}
        for ntype in h:
            n_id = self.node_dict[ntype]
            out_dict[ntype] = self.outs[n_id](h[ntype])
        return out_dict

    @torch.no_grad()
    def inference(self, graph, device):
        """
        整图推理，与 HGT 类似：
          - 每层都用 FullNeighborSampler，把结果写回 CPU
          - 层与层之间在 CPU 上中转
        """
        self.eval()
        # 1) 初始投影
        h = {}
        for ntype in graph.ntypes:
            n_id = self.node_dict[ntype]
            feat = graph.nodes[ntype].data["feat"].to(device)
            h[ntype] = F.gelu(self.adapt_ws[n_id](feat))

        # 2) 分层推理
        for layer_idx, layer in enumerate(self.layers):
            # 为本层输出准备 CPU 容器
            y = {
                ntype: torch.zeros(
                    graph.num_nodes(ntype), self.n_hid, device="cpu"
                )
                for ntype in graph.ntypes
            }

            sampler = MultiLayerFullNeighborSampler(1)
            dataloader = DataLoader(
                graph,
                {ntype: torch.arange(graph.num_nodes(ntype), dtype=torch.int32).to(device)
                 for ntype in graph.ntypes},
                sampler,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

            pbar = tqdm(dataloader, desc=f"Inference Layer {layer_idx+1}/{self.n_layers}", ncols=120)
            for input_nodes, output_nodes, blocks in pbar:
                b = blocks[0].to(device)

                # 提取输入节点特征
                h_input = {}
                for ntype in input_nodes:
                    # input_nodes[ntype] 是该 batch 的 src 全局 ID
                    h_input[ntype] = h[ntype][input_nodes[ntype].cpu()].to(device)

                # 前向
                h_new = layer(b, h_input)

                # 写回 CPU
                for ntype in h_new:
                    dst_nids = b.dstnodes[ntype].data[dgl.NID]
                    y[ntype][dst_nids] = h_new[ntype].cpu()

            # 更新 h
            h = y

        # 3) 最后输出层
        final_repr = {}
        for ntype in h:
            n_id = self.node_dict[ntype]
            final_repr[ntype] = self.outs[n_id](h[ntype].to(device))

        return final_repr
