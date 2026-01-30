# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
import graph_tool
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from ..substructure_dataset import SubstructureDataset
import torch.distributed as dist
import torch

import os
import os.path as osp
import pickle
from typing import Callable, List, Optional

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.io import fs


class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()
'''
class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):

        if not dist.is_initialized() or dist.get_rank() == 0:

            # super(MyZINC, self).process()

            print(" MyZINC.process() called")

            for split in ['train', 'val', 'test']:
                with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                    mols = pickle.load(f)

                indices = list(range(len(mols)))

                if self.subset:
                    with open(osp.join(self.raw_dir, f'{split}.index')) as f:
                        indices = [int(x) for x in f.read()[:-1].split(',')]

                pbar = tqdm(total=len(indices))
                pbar.set_description(f'Processing {split} dataset')

                data_list = []
                for idx in indices:
                    mol = mols[idx]

                    x = mol['atom_type'].to(torch.long).view(-1, 1)
                    y = mol['logP_SA_cycle_normalized'].to(torch.float)

                    adj = mol['bond_type']
                    edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                    edge_attr = adj[edge_index[0], edge_index[1]].to(torch.long)

                    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                y=y,global_idx=torch.tensor(idx))

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue

                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                    pbar.update(1)

                pbar.close()
                #  打印前几个样本的 global_idx 进行验证
                print(f"\n [{split}] 示例 global_idx:")
                for i in range(min(3, len(data_list))):
                    print(f"  sample[{i}].global_idx = {data_list[i].global_idx.item()}")

                self.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))
                # torch.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))
                #data, slices = self.collate(data_list)
                #self.save((data, slices), osp.join(self.processed_dir, f'{split}.pt'))
        if dist.is_initialized():

            dist.barrier()
'''
class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(" MyZINC.process() called")

            for split in ['train', 'val', 'test']:
                # 1. 加载 pickle 和 index
                with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                    mol_pairs = pickle.load(f)

                indices = list(range(len(mol_pairs)))
                if self.subset:
                    with open(osp.join(self.raw_dir, f'{split}.index')) as f:
                        indices = [int(x) for x in f.read().strip().split(',')]

                pbar = tqdm(total=len(indices))
                pbar.set_description(f'Processing {split} dataset')

                data_list = []

                # 2. 遍历每个药物对
                for idx in indices:
                    mol = mol_pairs[idx]

                    # --- 药物1 ---
                    x1 = mol['x1'].to(torch.float)
                    adj1 = mol['adj1']
                    edge_index1 = adj1.nonzero(as_tuple=False).t().contiguous()
                    bond_feats1 = mol['bond_feats1']  # shape [num_atoms, num_atoms, bond_feat_dim]
                    edge_attr1 = bond_feats1[edge_index1[0], edge_index1[1], :].to(torch.float)  # [num_edges, bond_feat_dim]

                    if edge_index1.shape[1] != edge_attr1.shape[0]:
                        print(f"[警告] edge_index1 与 edge_attr1 不匹配: {idx}")
                        print(f"  global_idx1: {global_idx1}")
                        print(f"  edge_index1.shape: {edge_index1.shape}")
                        print(f"  edge_attr1.shape:  {edge_attr1.shape}")

                    if edge_index1.numel() == 0:
                        print(f"Warning: edge_index1 is empty for molecule pair {idx} (global_idx1={global_idx1})")

                    # --- 药物2 ---
                    x2 = mol['x2'].to(torch.float)
                    adj2 = mol['adj2']
                    edge_index2 = adj2.nonzero(as_tuple=False).t().contiguous()
                    bond_feats2 = mol['bond_feats2']
                    edge_attr2 = bond_feats2[edge_index2[0], edge_index2[1], :].to(torch.float)

                    if edge_index2.shape[1] != edge_attr2.shape[0]:
                        print(f"[警告] edge_index2 与 edge_attr2 不匹配: {idx}")
                        print(f"  global_idx2: {global_idx2}")
                        print(f"  edge_index2.shape: {edge_index2.shape}")
                        print(f"  edge_attr2.shape:  {edge_attr2.shape}")

                    if edge_index2.numel() == 0:
                        print(f"Warning: edge_index2 is empty for molecule pair {idx} (global_idx2={global_idx2})")
                    # 标签和索引
                    #y = mol['y'].to(torch.float)  # shape = [1]
                    y = torch.tensor([mol['y']], dtype=torch.float)

                    global_idx1 = mol['global_idx1']
                    global_idx2 = mol['global_idx2']

                    data = Data(
                        x1=x1, edge_index1=edge_index1, edge_attr1=edge_attr1,
                        x2=x2, edge_index2=edge_index2, edge_attr2=edge_attr2,
                        y=y,
                        global_idx1=global_idx1,
                        global_idx2=global_idx2
                        #idx=torch.tensor([idx])  # optional: 图对的 id
                    )

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                    pbar.update(1)

                pbar.close()

                print(f"\n [{split}] 示例:")
                for i in range(min(3, len(data_list))):
                    y_val = data_list[i].y.view(-1).item()
                    print(f"  sample[{i}] → y={y_val}, "
                          f"global_idx1={data_list[i].global_idx1}, global_idx2={data_list[i].global_idx2}")
                self.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))

        if dist.is_initialized():
            dist.barrier()

class MYZINC_UNI(torch.utils.data.Dataset):
    def __init__(self,data_list):
        self.data_list=data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data=self.data_list[idx]
        return data

class Mydata(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(" MyZINC.process() called")

            for split in ['train', 'val', 'test']:
                # 1. 加载 pickle 和 index
                with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                    mol_pairs = pickle.load(f)

                indices = list(range(len(mol_pairs)))
                if self.subset:
                    with open(osp.join(self.raw_dir, f'{split}.index')) as f:
                        indices = [int(x) for x in f.read().strip().split(',')]

                pbar = tqdm(total=len(indices))
                pbar.set_description(f'Processing {split} dataset')

                data_list = []

                # 2. 遍历每个药物对
                for idx in indices:
                    mol = mol_pairs[idx]

                    # --- 药物1 ---
                    x1 = mol['x1'].to(torch.float)
                    adj1 = mol['adj1']
                    edge_index1 = adj1.nonzero(as_tuple=False).t().contiguous()
                    bond_feats1 = mol['bond_feats1']  # shape [num_atoms, num_atoms, bond_feat_dim]
                    edge_attr1 = bond_feats1[edge_index1[0], edge_index1[1], :].to(torch.float)  # [num_edges, bond_feat_dim]

                    if edge_index1.shape[1] != edge_attr1.shape[0]:
                        print(f"[警告] edge_index1 与 edge_attr1 不匹配: {idx}")
                        print(f"  global_idx1: {global_idx1}")
                        print(f"  edge_index1.shape: {edge_index1.shape}")
                        print(f"  edge_attr1.shape:  {edge_attr1.shape}")

                    if edge_index1.numel() == 0:
                        print(f"Warning: edge_index1 is empty for molecule pair {idx} (global_idx1={global_idx1})")

                    # --- 药物2 ---
                    x2 = mol['x2'].to(torch.float)
                    adj2 = mol['adj2']
                    edge_index2 = adj2.nonzero(as_tuple=False).t().contiguous()
                    bond_feats2 = mol['bond_feats2']
                    edge_attr2 = bond_feats2[edge_index2[0], edge_index2[1], :].to(torch.float)

                    if edge_index2.shape[1] != edge_attr2.shape[0]:
                        print(f"[警告] edge_index2 与 edge_attr2 不匹配: {idx}")
                        print(f"  global_idx2: {global_idx2}")
                        print(f"  edge_index2.shape: {edge_index2.shape}")
                        print(f"  edge_attr2.shape:  {edge_attr2.shape}")

                    if edge_index2.numel() == 0:
                        print(f"Warning: edge_index2 is empty for molecule pair {idx} (global_idx2={global_idx2})")
                    # 标签和索引
                    #y = mol['y'].to(torch.float)  # shape = [1]
                    y = torch.tensor([mol['y']], dtype=torch.float)

                    global_idx1 = mol['global_idx1']
                    global_idx2 = mol['global_idx2']

                    data = Data(
                        x1=x1, edge_index1=edge_index1, edge_attr1=edge_attr1,
                        x2=x2, edge_index2=edge_index2, edge_attr2=edge_attr2,
                        y=y,
                        global_idx1=global_idx1,
                        global_idx2=global_idx2
                        #idx=torch.tensor([idx])  # optional: 图对的 id
                    )

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                    pbar.update(1)

                pbar.close()

                print(f"\n [{split}] 示例:")
                for i in range(min(3, len(data_list))):
                    y_val = data_list[i].y.view(-1).item()
                    print(f"  sample[{i}] → y={y_val}, "
                          f"global_idx1={data_list[i].global_idx1}, global_idx2={data_list[i].global_idx2}")
                self.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))

        if dist.is_initialized():
            dist.barrier()

class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()




class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int,args=None,**kwargs) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None

        root = "dataset"


        if name == "zinc":

            inner_dataset = MyZINC(root=kwargs['data_dir'],subset=True)
            train_set = MyZINC(root=kwargs['data_dir'],subset=True, split="train")
            valid_set = MyZINC(root=kwargs['data_dir'],subset=True, split="val")
            test_set = MyZINC(root=kwargs['data_dir'],subset=True, split="test")
        if name == "mydata":

            inner_dataset = Mydata(root=kwargs['data_dir'],subset=True)
            train_set = Mydata(root=kwargs['data_dir'],subset=True, split="train")
            valid_set = Mydata(root=kwargs['data_dir'],subset=True, split="val")
            test_set = Mydata(root=kwargs['data_dir'],subset=True, split="test")

        elif name in ["CLUSTER","PATTERN"]:
            train_set = GNNBenchmarkDataset(root=kwargs['data_dir'], name=name, split='train')
            valid_set = GNNBenchmarkDataset(root=kwargs['data_dir'], name=name, split='val')
            test_set = GNNBenchmarkDataset(root=kwargs['data_dir'], name=name, split='test')

        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")

        if args['valid_on_test']:
            print("Validation on test")
            valid_set = test_set

        if train_set is not None:
            result= SubstructureDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                args['not_re_define'],
                args=args
                )
        else:
            result= (
                None
                if inner_dataset is None
                else SubstructureDataset(inner_dataset, seed)
            )

        return result
