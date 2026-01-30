# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
import joblib
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache
import pyximport
import torch.distributed as dist

pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
import sys
from tqdm import tqdm




def graph_data_modification_pair(data, sub1, sub2, adj1=None, adj2=None):
    for attr in ["x1", "x2", "edge_attr1", "edge_attr2", "edge_index1", "edge_index2"]:
        assert hasattr(data, attr), f"Missing attribute {attr} in pair data"

    # Patch each individual graph
    for i in [1, 2]:
        x = getattr(data, f"x{i}")
        if x is not None and x.dim() == 1:
            setattr(data, f"x{i}", x.unsqueeze(-1))
        edge_attr = getattr(data, f"edge_attr{i}")
        if edge_attr is not None and edge_attr.dim() == 1:
            setattr(data, f"edge_attr{i}", edge_attr.unsqueeze(-1))
    if data.y.dim() == 2:
        data.y = data.y.squeeze(1)

    # Remove unnecessary attrs if present
    for i in [1, 2]:
        if hasattr(data, f"edge_features{i}"):
            setattr(data, f"edge_attr{i}", getattr(data, f"edge_features{i}"))
            delattr(data, f"edge_features{i}")

    # Attach identifiers and sorted_adj
    data.identifiers1 = sub1
    data.identifiers2 = sub2
    data.sorted_adj1 = adj1
    data.sorted_adj2 = adj2

    return data
def encode_token_pair_tensor(data, atom_id_max, edge_id_max):
    for i in [1, 2]:
        x = getattr(data, f"x{i}")
        edge_index = getattr(data, f"edge_index{i}")
        edge_attr = getattr(data, f"edge_attr{i}")
        identifiers = getattr(data, f"identifiers{i}")

        cur_id = x.shape[0]
        feat_dim = x.shape[1]  # 新增，支持高维原子特征
        _, index = np.unique(identifiers[0, :], return_index=True)
        index = np.sort(index)
        token_ids = identifiers[2, index].unsqueeze(1) + atom_id_max + 1

        # ===============  拼接token节点的高维特征 ===============
        num_new_tokens = token_ids.shape[0]
        # 推荐用全-1，也可用全0或其它自定义
        token_feats = -1 * torch.ones((num_new_tokens, feat_dim), dtype=x.dtype, device=x.device)
        x_new = torch.cat([x, token_feats], 0)
        setattr(data, f"x{i}", x_new)

        # ===============  处理edge_index扩展 ===============
        edges = torch.cat([
            identifiers[0, :].unsqueeze(0) + cur_id,
            identifiers[1, :].unsqueeze(0)
        ], 0).long()
        edge_index_new = torch.cat([edge_index, edges, edges[[1, 0], :]], 1)
        setattr(data, f"edge_index{i}", edge_index_new)

        # =============== 处理edge_attr扩展 ===============
        if edge_attr is not None:
            edge_attr_dim = edge_attr.shape[1]
            # token边特征可全-1或全0
            edge_attr_token = -1 * torch.ones((edges.shape[1], edge_attr_dim), dtype=edge_attr.dtype,
                                              device=edge_attr.device)
            edge_attr_new = torch.cat([edge_attr, edge_attr_token, edge_attr_token], 0)
            setattr(data, f"edge_attr{i}", edge_attr_new)

        # =============== . mask扩展 ===============
        sub_adj_mask = torch.ones([x_new.shape[0], 1]).long()
        sub_adj_mask[0:cur_id] = 0
        setattr(data, f"sub_adj_mask{i}", sub_adj_mask)

    return data
def encode_token_pair_tensor_with_adj(data, local_attention_on_substructures):
    for i in [1, 2]:
        x = getattr(data, f"x{i}")
        edge_index = getattr(data, f"edge_index{i}")
        edge_attr = getattr(data, f"edge_attr{i}")
        identifiers = getattr(data, f"identifiers{i}")
        sorted_adj = getattr(data, f"sorted_adj{i}")

        cur_id = x.shape[0]
        feat_dim = x.shape[1]
        _, index = np.unique(identifiers[0, :], return_index=True)
        index = np.sort(index)

        # === 1. token节点特征，全部为-1 ===
        num_new_tokens = index.shape[0]
        token_feats = -1 * torch.ones((num_new_tokens, feat_dim), dtype=x.dtype, device=x.device)

        # === 2. 拼接 token 节点到 x ===
        x_new = torch.cat([x, token_feats], 0)
        setattr(data, f"x{i}", x_new)

        # === 3. sorted_adj 扩展 ===
        sorted_adj = torch.cat(
            [torch.zeros([x.size(0), *sorted_adj.shape[1:]], dtype=sorted_adj.dtype, device=sorted_adj.device),
             sorted_adj], 0)
        setattr(data, f"sorted_adj{i}", sorted_adj)

        # === 4. mask扩展 ===
        sub_adj_mask = torch.ones([x_new.shape[0], 1], dtype=torch.long, device=x.device)
        sub_adj_mask[0:cur_id] = 0
        setattr(data, f"sub_adj_mask{i}", sub_adj_mask)

        # === 5. edge_index扩展 ===
        edges = torch.cat([
            identifiers[0, :].unsqueeze(0) + cur_id,
            identifiers[1, :].unsqueeze(0)
        ], 0).long()
        edge_index_new = torch.cat([edge_index, edges, edges[[1, 0], :]], 1)
        setattr(data, f"edge_index{i}", edge_index_new)


        if edge_attr is not None:
            edge_attr_dim = edge_attr.shape[1]
            num_new_edges = edges.shape[1]  # 新增token边的数量（单向）

            edge_attr_token = -1 * torch.ones((num_new_edges, edge_attr_dim), dtype=edge_attr.dtype,
                                          device=edge_attr.device)

            edge_attr_new = torch.cat([edge_attr, edge_attr_token, edge_attr_token], 0)
            setattr(data, f"edge_attr{i}", edge_attr_new)
    return data
def preprocess_item_pair(data, local_attention_on_substructures=False, continuous_feature=False):
    max_dist_const = 510  # 与单图版本保持一致
    substructure_dist_const = max_dist_const - 1

    for i in [1, 2]:
        # 获取当前图的属性
        x = getattr(data, f"x{i}")
        edge_index = getattr(data, f"edge_index{i}")
        edge_attr = getattr(data, f"edge_attr{i}")
        cur_id = getattr(data, f"cur_id{i}") if hasattr(data, f"cur_id{i}") else x.size(0)


        N = x.size(0)

        setattr(data, f"x{i}", x)  # [N, 64]


        adj_matrix = torch.zeros([N, N], dtype=torch.bool)
        adj_matrix[edge_index[0, :], edge_index[1, :]] = True


        if local_attention_on_substructures:
            adj = adj_matrix[0:cur_id, 0:cur_id]
        else:
            adj = adj_matrix[0:, 0:]


        if edge_attr is None:
            attn_edge_type = -1 * torch.ones([N, N, 1], dtype=torch.float)
        else:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr[:, None]
            feat_dim = edge_attr.size(-1)

            attn_edge_type = torch.zeros([N, N, feat_dim], dtype=torch.float)
            attn_edge_type[edge_index[0], edge_index[1]] = edge_attr  # 每条边赋值多维特征


        if local_attention_on_substructures:
            adj_matrix = adj_matrix.float()
            adj_matrix[0:cur_id, 0:cur_id] = 1
            attn_bias = 1 - adj_matrix
            attn_bias[attn_bias > 0] = float('-inf')
            attn_bias_res = torch.zeros([N + 1, N + 1], dtype=torch.float)
            attn_bias_res[1:, 1:] = attn_bias
            attn_bias = attn_bias_res
        else:
            attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)


        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        max_dist = np.amax(shortest_path_result)
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        spatial_pos = torch.from_numpy(shortest_path_result).long()


        if local_attention_on_substructures:
            edge_input_new = (-1 * np.ones([N, N, edge_input.shape[2], edge_input.shape[3]])).astype(np.int64)
            edge_input_new[0:cur_id, 0:cur_id, :, :] = edge_input
            edge_input = edge_input_new

            spatial_pos_new = (max_dist_const * torch.ones(N, N)).long()
            spatial_pos_new[0:cur_id, 0:cur_id] = spatial_pos
            spatial_pos = spatial_pos_new


        setattr(data, f"attn_bias{i}", attn_bias)
        setattr(data, f"attn_edge_type{i}", torch.tensor(attn_edge_type))
        setattr(data, f"spatial_pos{i}", spatial_pos)
        setattr(data, f"in_degree{i}", adj_matrix.long().sum(dim=1).view(-1))
        setattr(data, f"out_degree{i}", adj_matrix.long().sum(dim=1).view(-1))  # 无向图相同
        setattr(data, f"edge_input{i}", torch.from_numpy(edge_input).float())  # 维度[N, N, max_dist, feat_dim]


    return data