import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class GATLayer(nn.Module):
    """
    多头 GAT 版本的单层聚合：
      - 对于 (srctype, etype, dsttype) 的每条边，用源/目的节点的投影特征计算注意力分数
      - multi-head -> edge_softmax -> sum 聚合 -> 残差、LayerNorm（可选）
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        node_dict,      # { ntype -> idx }
        edge_dict,      # { etype -> idx }, 如果想对不同 etype 使用不同参数也可以
        n_heads,
        dropout=0.2,
        use_norm=False,
        negative_slope=0.2  # leaky_relu 的斜率
    ):
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.num_types = len(node_dict)
        self.num_relations = len(edge_dict)
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.use_norm = use_norm
        self.negative_slope = negative_slope

        # 对每种节点类型的线性变换 (投影到 n_heads * d_k)
        # 注意: 这里写成 out_dim，实际是 n_heads * d_k，若 out_dim = n_heads * d_k 则等价
        self.fc = nn.ModuleList()
        self.att_l = nn.ParameterList()  
        self.att_r = nn.ParameterList()
        self.norms = nn.ModuleList()

        for _ in range(self.num_types):
            self.fc.append(nn.Linear(in_dim, out_dim, bias=False))
            # GAT 常规做法: a_l, a_r 形状为 (n_heads, d_k)
            self.att_l.append(nn.Parameter(torch.Tensor(self.n_heads, self.d_k)))
            self.att_r.append(nn.Parameter(torch.Tensor(self.n_heads, self.d_k)))

            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))
            else:
                self.norms.append(None)

        # 残差系数
        self.skip = nn.Parameter(torch.ones(self.num_types))
        self.drop = nn.Dropout(dropout)

        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.num_types):
            nn.init.xavier_uniform_(self.fc[i].weight)
            nn.init.xavier_uniform_(self.att_l[i])
            nn.init.xavier_uniform_(self.att_r[i])

            if self.norms[i] is not None:
                nn.init.constant_(self.norms[i].weight, 1)
                nn.init.constant_(self.norms[i].bias, 0)

    def forward(self, block, h_dict):
        """
        block: dgl.Block
        h_dict: { ntype: Tensor }，形如 [#src_nodes_ntype, in_dim]
        返回: { ntype: Tensor }，更新后的目标节点特征
        """
        with block.local_scope():
            # 1) 对每条边 (srctype, etype, dsttype) 计算注意力
            for srctype, etype, dsttype in block.canonical_etypes:
                sub_block = block[srctype, etype, dsttype]
                e_id = self.edge_dict[etype]

                # 获取源、目标节点在 block 内的局部ID
                src_ids = sub_block.srcnodes(srctype)
                dst_ids = sub_block.dstnodes(dsttype)

                # 根据节点类型，选取相应的投影 fc & att 参数
                s_id = self.node_dict[srctype]
                d_id = self.node_dict[dsttype]
                
                # 投影
                h_src = self.fc[s_id](h_dict[srctype][src_ids])  # [num_src, out_dim]
                h_dst = self.fc[d_id](h_dict[dsttype][dst_ids])  # [num_dst, out_dim]

                # 形状变为 (num_src, n_heads, d_k)
                h_src = h_src.view(-1, self.n_heads, self.d_k)
                h_dst = h_dst.view(-1, self.n_heads, self.d_k)

                # 计算注意力分数
                # e_{ij} = leaky_relu( (h_src * att_l[s_id]) + (h_dst * att_r[d_id]) )
                # 注意: (h_src * att_l[s_id]) 是逐元素相乘 -> shape (num_src, n_heads, d_k)
                # 然后再对 dim=2 做 sum -> shape (num_src, n_heads)
                alpha_src = (h_src * self.att_l[s_id]).sum(dim=-1)  # [num_src, n_heads]
                alpha_dst = (h_dst * self.att_r[d_id]).sum(dim=-1)  # [num_dst, n_heads]

                # 存入 sub_block
                sub_block.srcdata["alpha_src"] = alpha_src
                sub_block.dstdata["alpha_dst"] = alpha_dst
                sub_block.srcdata["h_src"] = h_src  # 消息
                # 用 dgl 内置函数计算 e_ij = alpha_src + alpha_dst
                # 先把 alpha_src 传给边，再加上 alpha_dst
                sub_block.apply_edges(
                    fn.u_add_v("alpha_src", "alpha_dst", "e")
                )
                # e 形状 [num_edges, n_heads]
                e = sub_block.edata["e"]
                e = F.leaky_relu(e, negative_slope=self.negative_slope)
                # 做 softmax
                alpha = edge_softmax(sub_block, e, norm_by="dst")  # [num_edges, n_heads]
                sub_block.edata["alpha"] = alpha.unsqueeze(-1)     # 扩展维度便于后续乘消息

            # 2) 消息聚合
            # 对每个 etype 定义消息、聚合函数
            update_dict = {}
            for srctype, etype, dsttype in block.canonical_etypes:
                sub_block = block[srctype, etype, dsttype]
                # e_id = self.edge_dict[etype]  # 如果你需要区分 etype，可再拿 e_id
                update_dict[(srctype, etype, dsttype)] = (
                    fn.u_mul_e("h_src", "alpha", "m"),  # 消息函数
                    fn.sum("m", "t"),                  # 聚合函数
                )

            block.multi_update_all(update_dict, cross_reducer="mean")

            # 3) 残差 + (可选)LayerNorm
            new_h = {}
            for ntype in block.ntypes:
                n_id = self.node_dict[ntype]
                alpha = torch.sigmoid(self.skip[n_id])  # 残差系数

                if "t" in block.nodes[ntype].data:
                    # t 形状 [num_dst_nodes_ntype, n_heads, d_k]
                    t = block.nodes[ntype].data["t"]
                    t = t.view(-1, self.out_dim)
                else:
                    t = torch.zeros(
                        (block.num_dst_nodes(ntype), self.out_dim),
                        device=h_dict[ntype].device
                    )

                # 线性变换前可以加 dropout
                out = self.drop(t)

                # 残差： out = alpha * out + (1-alpha) * h_dst
                dst_ids = block.dstnodes(ntype)
                h_dst = h_dict[ntype][dst_ids]
                # 同样 reshape
                # (注：若 in_dim != out_dim，可以改成 out += residual_linear(h_dst))
                out = out * alpha + h_dst * (1 - alpha)

                # LayerNorm
                if self.norms[n_id] is not None:
                    out = self.norms[n_id](out)

                new_h[ntype] = out

            return new_h

class GAT(nn.Module):
    """
    多层 GAT 模型，借助上面的 GATLayer 做多头注意力聚合。
    与 HAN 类似的结构：先做 input projection，再堆叠 GATLayer，最后输出层。
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
        use_norm=True
    ):
        super(GAT, self).__init__()
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_hid = n_hid
        self.n_out = n_out
        self.n_layers = n_layers

        # 1) 针对每种节点类型的输入特征 => n_hid
        self.adapt_ws = nn.ModuleList()
        for ntype in G.ntypes:
            in_dim = G.nodes[ntype].data["feat"].shape[-1]
            self.adapt_ws.append(nn.Linear(in_dim, n_hid))

        # 2) 叠加 n_layers 个 GATLayer
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                GATLayer(
                    in_dim=n_hid,
                    out_dim=n_hid,
                    node_dict=node_dict,
                    edge_dict=edge_dict,
                    n_heads=n_heads,
                    dropout=0.2,
                    use_norm=use_norm,
                )
            )

        # 3) 输出层
        self.outs = nn.ModuleList()
        for _ in G.ntypes:
            self.outs.append(nn.Linear(n_hid, n_out))

    def forward(self, blocks, node_features):
        """
        blocks: List[dgl.Block]，长度 = n_layers
        node_features: { ntype: Tensor }，形如 [batch_size_ntype, in_dim]
        返回: { ntype: Tensor }，每种目标节点的输出 (维度 n_out)
        """
        # 先将输入特征投影到 n_hid
        for ntype in node_features:
            idx = self.node_dict[ntype]
            x = node_features[ntype]
            node_features[ntype] = F.gelu(self.adapt_ws[idx](x))

        # 逐层执行 GATLayer
        h = node_features
        for layer_idx, layer in enumerate(self.layers):
            h = layer(blocks[layer_idx], h)

        # 最终输出
        out_dict = {}
        for i, ntype in enumerate(h.keys()):
            out_dict[ntype] = self.outs[i](h[ntype])
        return out_dict

    @torch.no_grad()
    def inference(self, graph, device):
        """
        整图推理，与 HAN.inference 类似，先做 input projection，
        然后对每层用 FullNeighborSampler 求子图批量推理。
        """
        self.eval()
        # 1) 初始投影
        h = {}
        for i, ntype in enumerate(graph.ntypes):
            feat = graph.nodes[ntype].data["feat"].to(device)
            h[ntype] = F.gelu(self.adapt_ws[i](feat))

        # 2) 分层推理
        for layer_idx, layer in enumerate(self.layers):
            # 为本层输出准备 CPU 容器
            y = {
                ntype: torch.zeros(
                    graph.num_nodes(ntype), self.n_hid, device="cpu"
                )
                for ntype in graph.ntypes
            }
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                graph,
                {ntype: torch.arange(graph.num_nodes(ntype), dtype=torch.int32).to(device)
                 for ntype in graph.ntypes},
                sampler,
                batch_size=1024,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

            for input_nodes, output_nodes, blocks in dataloader:
                b = blocks[0].to(device)
                h_input = {}
                for ntype in input_nodes:
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
        for i, ntype in enumerate(h.keys()):
            final_repr[ntype] = self.outs[i](h[ntype].to(device))

        return final_repr
