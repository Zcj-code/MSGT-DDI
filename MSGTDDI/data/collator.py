# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def build_padding_mask(lengths, max_length):
    """Build a boolean key-padding mask from the true node counts.

    False means a real node; True means a padded position.
    The mask must never be inferred from an atom-feature column.
    """
    mask = torch.ones((len(lengths), max_length), dtype=torch.bool)
    for batch_index, length in enumerate(lengths):
        length = int(length)
        if length < 0 or length > max_length:
            raise ValueError(
                f"Invalid node count {length}; expected 0 <= length <= {max_length}."
            )
        mask[batch_index, :length] = False
    return mask

def pad_1d_unsqueeze(x, padlen, pad_value=0):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_full([padlen], pad_value, dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen, pad_value=0):
    xlen = x.size(0)
    shape = list(x.shape)
    shape[0] = padlen
    new_x = x.new_full(shape, pad_value, dtype=x.dtype)
    new_x[:xlen] = x
    return new_x.unsqueeze(0)

def pad_sub_adjs_unsqueeze(x, padlen, pad_value=0):
    xlen = x.size(0)
    shape = list(x.shape)
    shape[0] = padlen
    new_x = x.new_full(shape, pad_value, dtype=x.dtype)
    new_x[:xlen] = x
    return new_x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen, pad_value=float("-inf")):
    xlen = x.size(0)
    new_x = x.new_full([padlen, padlen], pad_value, dtype=x.dtype)
    new_x[:xlen, :xlen] = x
    if pad_value == float("-inf"):
        new_x[xlen:, :xlen] = 0  # 可选，避免nan
    return new_x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen, pad_value=0):
    xlen = x.size(0)
    shape = list(x.shape)
    shape[0] = padlen
    shape[1] = padlen
    new_x = x.new_full(shape, pad_value, dtype=x.dtype)
    new_x[:xlen, :xlen] = x
    return new_x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen, pad_value=0):
    xlen = x.size(0)
    new_x = x.new_full([padlen, padlen], pad_value, dtype=x.dtype)
    new_x[:xlen, :xlen] = x
    return new_x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3, pad_value=0):
    shape = list(x.shape)
    shape[0], shape[1], shape[2] = padlen1, padlen2, padlen3
    new_x = x.new_full(shape, pad_value, dtype=x.dtype)
    new_x[:x.shape[0], :x.shape[1], :x.shape[2]] = x
    return new_x.unsqueeze(0)

def pad_3d_sequences(xs, padlen1, padlen2, padlen3, pad_value=0):
    feat_dim = xs[0].shape[-1]
    result = xs[0].new_full([len(xs), padlen1, padlen2, padlen3, feat_dim], pad_value, dtype=xs[0].dtype)
    for i, x in enumerate(xs):
        result[i, :x.shape[0], :x.shape[1], :x.shape[2], :] = x
    return result

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [
        (
            item.idx,
            item.attn_bias,
            item.attn_edge_type,
            item.spatial_pos,
            item.in_degree,
            item.out_degree,
            item.x,
            item.edge_input[:, :, :multi_hop_max_dist, :],
            item.y,
        )
        for item in items
    ]
    (
        idxs,
        attn_biases,
        attn_edge_types,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_inputs,
        ys,
    ) = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    padding_mask = build_padding_mask([x.size(0) for x in xs], max_node_num)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat(
        [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
    )
    attn_bias = torch.cat(
        [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
    )
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
    )
    spatial_pos = torch.cat(
        [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
    )
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,  # for undirected graph
        x=x,
        edge_input=edge_input,
        padding_mask=padding_mask,
        y=y,
    )



def collator_adj_target_pair(items, max_node=512, node_level_task=False):
    original_count = len(items)

    filtered_items = []
    for i, item in enumerate(items):
        if item is None:
            print(f" target移除空样本 (索引 {i})")
            continue
        if item.x1.size(0) > max_node or item.x2.size(0) > max_node:
            print(
                f" 移target除超限样本 (索引 {i}): x1节点数={item.x1.size(0)}, x2节点数={item.x2.size(0)}, 最大允许={max_node}，x1id={item.global_idx1}, x2id={item.global_idx2}")
            continue
        filtered_items.append(item)

    # 在过滤后打印结果
    removed_count = original_count - len(filtered_items)
    if removed_count > 0:
        print(f" target过滤后样本总数: {len(filtered_items)}, 移除了 {removed_count} 个样本")
        print(f" target保留率: {len(filtered_items) / original_count:.1%}")

    items = filtered_items


    samples = [item.y for item in items]

    if node_level_task:
        return collator_node_label(samples)
    else:

        return torch.stack(samples, dim=0)


def collator_adj_pair(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    original_count = len(items)

    filtered_items = []
    for i, item in enumerate(items):
        if item is None:
            print(f" 移除空样本 (索引 {i})")
            continue
        if item.x1.size(0) > max_node or item.x2.size(0) > max_node:
            print(
                f" 移除超限样本 (索引 {i}): x1节点数={item.x1.size(0)}, x2节点数={item.x2.size(0)}, 最大允许={max_node}，x1id={item.global_idx1}, x2id={item.global_idx2}")
            continue
        filtered_items.append(item)



    items = filtered_items

    batch = []
    for item in items:
        cur_id1 = getattr(item, "cur_id1", int((item.sub_adj_mask1 == 0).sum().item()) if item.sub_adj_mask1 is not None else item.x1.size(0))
        cur_id2 = getattr(item, "cur_id2", int((item.sub_adj_mask2 == 0).sum().item()) if item.sub_adj_mask2 is not None else item.x2.size(0))
        num_subtokens1 = getattr(item, "num_subtokens1", item.x1.size(0) - cur_id1)
        num_subtokens2 = getattr(item, "num_subtokens2", item.x2.size(0) - cur_id2)
        batch.append((
            item.idx,
            item.global_idx1,
            item.global_idx2,
            item.attn_bias1,
            item.attn_bias2,
            item.attn_edge_type1,
            item.attn_edge_type2,
            item.spatial_pos1,
            item.spatial_pos2,
            item.in_degree1,
            item.in_degree2,
            item.out_degree1,
            item.out_degree2,
            item.x1,
            item.x2,
            item.edge_input1[:, :, :multi_hop_max_dist, :],
            item.edge_input2[:, :, :multi_hop_max_dist, :],
            item.y,
            item.sorted_adj1,
            item.sorted_adj2,
            item.sub_adj_mask1,
            item.sub_adj_mask2,
            cur_id1,
            cur_id2,
            num_subtokens1,
            num_subtokens2,
            getattr(item, "identifiers1", None),
            getattr(item, "identifiers2", None),
        ))

    (
        idxs, gidx1s, gidx2s,
        attn_biases1, attn_biases2,
        edge_types1, edge_types2,
        spos1, spos2,
        indeg1, indeg2, outdeg1, outdeg2,
        xs1, xs2, edge_inputs1, edge_inputs2,
        ys, adj1s, adj2s, mask1s, mask2s,
        cur_ids1, cur_ids2, num_subtokens1, num_subtokens2,
        identifiers1, identifiers2
    ) = zip(*batch)

    for idx in range(len(attn_biases1)):
        attn_biases1[idx][1:, 1:][spos1[idx] >= spatial_pos_max] = float("-inf")
        attn_biases2[idx][1:, 1:][spos2[idx] >= spatial_pos_max] = float("-inf")

    # 统一 max node
    max_node_num1 = max(x.size(0) for x in xs1)
    max_node_num2 = max(x.size(0) for x in xs2)
    max_node_num = max(max_node_num1, max_node_num2)

    # Explicit padding masks. False = real atom/substructure token; True = padding.
    # x1 and x2 share max_node_num, but their true lengths can differ.
    padding_mask1 = build_padding_mask([x.size(0) for x in xs1], max_node_num)
    padding_mask2 = build_padding_mask([x.size(0) for x in xs2], max_node_num)

    max_dist1 = max(e.size(-2) for e in edge_inputs1)
    max_dist2 = max(e.size(-2) for e in edge_inputs2)
    max_dist = max(max_dist1, max_dist2)

    # Padding（使用你已有的 pad 函数）
    x1 = torch.cat([pad_2d_unsqueeze(x, max_node_num) for x in xs1])
    x2 = torch.cat([pad_2d_unsqueeze(x, max_node_num) for x in xs2])

    x1 = x1.float()
    x2 = x2.float()

    edge_input1 = pad_3d_sequences(edge_inputs1, max_node_num, max_node_num, max_dist)
    edge_input2 = pad_3d_sequences(edge_inputs2, max_node_num, max_node_num, max_dist)

    attn_bias1 = torch.cat([pad_attn_bias_unsqueeze(b, max_node_num + 1) for b in attn_biases1])
    attn_bias2 = torch.cat([pad_attn_bias_unsqueeze(b, max_node_num + 1) for b in attn_biases2])

    attn_edge_type1 = torch.cat([pad_edge_type_unsqueeze(e, max_node_num) for e in edge_types1])
    attn_edge_type2 = torch.cat([pad_edge_type_unsqueeze(e, max_node_num) for e in edge_types2])

    spatial_pos1 = torch.cat([pad_spatial_pos_unsqueeze(s, max_node_num) for s in spos1])
    spatial_pos2 = torch.cat([pad_spatial_pos_unsqueeze(s, max_node_num) for s in spos2])

    in_degree1 = torch.cat([pad_1d_unsqueeze(d, max_node_num) for d in indeg1])
    in_degree2 = torch.cat([pad_1d_unsqueeze(d, max_node_num) for d in indeg2])

    y = torch.cat(ys)

    sorted_adj1 = torch.cat([pad_sub_adjs_unsqueeze(adj, max_node_num) for adj in adj1s]) if adj1s[0] is not None else None
    sorted_adj2 = torch.cat([pad_sub_adjs_unsqueeze(adj, max_node_num) for adj in adj2s]) if adj2s[0] is not None else None

    sub_adj_mask1 = torch.cat([pad_2d_unsqueeze(m, max_node_num) for m in mask1s]) if mask1s[0] is not None else None
    sub_adj_mask2 = torch.cat([pad_2d_unsqueeze(m, max_node_num) for m in mask2s]) if mask2s[0] is not None else None

    return dict(
        idx=torch.LongTensor(idxs),
        global_idx1=gidx1s,
        global_idx2=gidx2s,

        x1=x1,
        x2=x2,
        padding_mask1=padding_mask1,
        padding_mask2=padding_mask2,

        edge_input1=edge_input1,
        edge_input2=edge_input2,

        attn_bias1=attn_bias1,
        attn_bias2=attn_bias2,

        attn_edge_type1=attn_edge_type1,
        attn_edge_type2=attn_edge_type2,

        spatial_pos1=spatial_pos1,
        spatial_pos2=spatial_pos2,

        in_degree1=in_degree1,
        in_degree2=in_degree2,

        out_degree1=in_degree1,
        out_degree2=in_degree2,

        y=y,

        sorted_adj1=sorted_adj1,
        sorted_adj2=sorted_adj2,
        sub_adj_mask1=sub_adj_mask1,
        sub_adj_mask2=sub_adj_mask2,
        cur_id1=torch.LongTensor(cur_ids1),
        cur_id2=torch.LongTensor(cur_ids2),
        num_atoms1=torch.LongTensor(cur_ids1),
        num_atoms2=torch.LongTensor(cur_ids2),
        num_subtokens1=torch.LongTensor(num_subtokens1),
        num_subtokens2=torch.LongTensor(num_subtokens2),
        identifiers1=identifiers1,
        identifiers2=identifiers2,
    )
#



def collatorcontrust(datas, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    datas_data=[a_data.data for a_data in datas]
    datas_data1 = [a_data.data1 for a_data in datas]
    datas_data2 = [a_data.data2 for a_data in datas]
    result=[]
    for items in [datas_data,datas_data1,datas_data2]:
        result.append(collator(items, max_node=max_node, multi_hop_max_dist=multi_hop_max_dist, spatial_pos_max=spatial_pos_max))
    res_dic={}
    res_dic['data']=result[0]
    res_dic['data1'] = result[1]
    res_dic['data2'] = result[2]
    return res_dic

