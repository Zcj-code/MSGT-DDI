# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import warnings
import contextlib
from dataclasses import dataclass, field
from omegaconf import II, open_dict, OmegaConf
import importlib
from fairseq import metrics, search, tokenizer, utils
import numpy as np
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
from fairseq.tasks import FairseqDataclass, FairseqTask, register_task
from fairseq.dataclass.utils import gen_parser_from_dataclass
from ..data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    GraphormerDataset,
    EpochShuffleDataset, GraphormerDDIDataset, BatchedDataDataset_DDISubstructure,
)

import torch
from fairseq.optim.amp_optimizer import AMPOptimizer
import math

from ..data import DATASET_REGISTRY
import sys
import os

logger = logging.getLogger(__name__)
#
# @dataclass

class GraphPredictionConfig(FairseqDataclass):
    dataset_name: str = field(
        default="zinc",
        metadata={"help": "name of the dataset"},
    )

    num_classes: int = field(
        default=-1,
        metadata={"help": "number of classes or regression targets"},
    )

    max_nodes: int = field(
        default=203,
        metadata={"help": "max nodes per graph"},
    )

    dataset_source: str = field(
        default="pyg",
        metadata={"help": "source of graph dataset, can be: pyg, dgl, ogb, smiles"},
    )

    num_atoms: int = field(
        default=10 * 9,
        metadata={"help": "number of atom types in the graph"},
    )

    num_edges: int = field(
        default=10 * 3,
        metadata={"help": "number of edge types in the graph"},
    )

    num_in_degree: int = field(
        default=512,
        metadata={"help": "number of in degree types in the graph"},
    )

    num_out_degree: int = field(
        default=512,
        metadata={"help": "number of out degree types in the graph"},
    )

    num_spatial: int = field(
        default=512,
        metadata={"help": "number of spatial types in the graph"},
    )

    num_edge_dis: int = field(
        default=128,
        metadata={"help": "number of edge dis types in the graph"},
    )

    multi_hop_max_dist: int = field(
        default=5,
        metadata={"help": "max distance of multi-hop edges"},
    )

    spatial_pos_max: int = field(
        default=1024,
        metadata={"help": "max distance of multi-hop edges"},
    )

    edge_type: str = field(
        default="multi_hop",
        metadata={"help": "edge type in the graph"},
    )

    seed: int = II("common.seed")

    pretrained_model_name: str = field(
        default="none",
        metadata={"help": "name of used pretrained model"},
    )

    load_pretrained_model_output_layer: bool = field(
        default=False,
        metadata={"help": "whether to load the output layer of pretrained model"},
    )

    train_epoch_shuffle: bool = field(
        default=False,
        metadata={"help": "whether to shuffle the dataset at each epoch"},
    )

    user_data_dir: str = field(
        default="",
        metadata={"help": "path to the module of user-defined dataset"},
    )

    data_dir: str = field(default="dataset", metadata={"help": "path to data"})

    not_re_define: bool = field(
        default=False,
    )

    valid_on_test: bool = field(default=False)

    node_level_task: bool = field(default=False)

    continuous_feature: bool = field(default=False)

    multi_label: int = field(default=0)

    fusion: str = field(default='None')

    reattention: bool = field(default=False)

@register_task("graph_prediction", dataclass=GraphPredictionConfig)
class GraphPredictionTask(FairseqTask):
    """
    Graph prediction (classification or regression) task.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.user_data_dir != "":
            print("user_data_dir != """)
            self.__import_user_defined_datasets(cfg.user_data_dir)
            if cfg.dataset_name in DATASET_REGISTRY:
                dataset_dict = DATASET_REGISTRY[cfg.dataset_name]
                self.dm = GraphormerDataset(
                    dataset=dataset_dict["dataset"],
                    dataset_source=dataset_dict["source"],
                    train_idx=dataset_dict["train_idx"],
                    valid_idx=dataset_dict["valid_idx"],
                    test_idx=dataset_dict["test_idx"],
                    args=cfg,
                    seed=cfg.seed)
            else:
                raise ValueError(f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}")
        else:
            kwargs={}
            kwargs['data_dir']=cfg.data_dir
            self.dm = GraphormerDataset(
                dataset_spec=cfg.dataset_name,
                dataset_source=cfg.dataset_source,
                seed=cfg.seed,
                args=cfg,
                **kwargs
            )


    def __import_user_defined_datasets(self, dataset_dir):
        dataset_dir = dataset_dir.strip("/")
        module_parent, module_name = os.path.split(dataset_dir)
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        for file in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, file)
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
            ):
                task_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(module_name + "." + task_name)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test

        batched_data = BatchedDataDataset(
            batched_data,
            max_node=self.max_nodes(),
            multi_hop_max_dist=self.cfg.multi_hop_max_dist,
            spatial_pos_max=self.cfg.spatial_pos_max,
        )

        data_sizes = np.array([self.max_nodes()] * len(batched_data))

        target = TargetDataset(batched_data,self.cfg.node_level_task)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset, size=len(dataset), seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        model = models.build_model(cfg, self)


        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None





#下面两个类是ddi预测相关类

@dataclass
class GraphPredictionDDIConfig(GraphPredictionConfig):
    # ===== 显式继承 GraphPredictionConfig 的字段（确保都进入 task 配置命名空间） =====
    dataset_name: str = field(default="zinc", metadata={"help": "name of the dataset"})  # 数据集名称（如 zinc）
    num_classes: int = field(default=1, metadata={"help": "number of classes or regression targets"})  # 类别数或回归目标维度
    max_nodes: int = field(default=220, metadata={"help": "max nodes per graph"})  # 单图最大节点数（用于截断/填充和张量上限）
    dataset_source: str = field(default="pyg", metadata={
        "help": "source of graph dataset, can be: pyg, dgl, ogb, smiles"})  # 数据集来源：pyg/dgl/ogb/smiles 等
    num_atoms: int = field(default=10 * 9, metadata={"help": "number of atom types in the graph"})  # 原子类型数量（节点类型种类）
    num_edges: int = field(default=10 * 3, metadata={"help": "number of edge types in the graph"})  # 边类型数量
    num_in_degree: int = field(default=512, metadata={"help": "number of in degree types in the graph"})  # 入度离散桶数量
    num_out_degree: int = field(default=512, metadata={"help": "number of out degree types in the graph"})  # 出度离散桶数量
    num_spatial: int = field(default=512, metadata={"help": "number of spatial types in the graph"})  # 空间/距离编码桶数量
    num_edge_dis: int = field(default=128, metadata={"help": "number of edge dis types in the graph"})  # 边距离离散桶数量
    multi_hop_max_dist: int = field(default=5, metadata={"help": "max distance of multi-hop edges"})  # 多跳最远距离（用于多跳边构图）
    spatial_pos_max: int = field(default=1024, metadata={"help": "max distance of multi-hop edges"})  # 位置编码/空间距离的最大截断
    edge_type: str = field(default="multi_hop", metadata={"help": "edge type in the graph"})  # 使用的边类型方案，如 multi_hop
    seed: int = II("common.seed")  # 随机种子（从 common.seed 注入）
    pretrained_model_name: str = field(default="none", metadata={"help": "name of used pretrained model"})  # 预训练模型名（若用）
    load_pretrained_model_output_layer: bool = field(default=False, metadata={
        "help": "whether to load the output layer of pretrained model"})  # 是否加载预训练输出层
    train_epoch_shuffle: bool = field(default=False, metadata={
        "help": "whether to shuffle the dataset at each epoch"})  # 每个 epoch 是否打乱样本
    user_data_dir: str = field(default="", metadata={
        "help": "path to the module of user-defined dataset"})  # 自定义数据集 Python 包路径（会 import）
    data_dir: str = field(default="dataset", metadata={"help": "path to data"})  # 数据根目录
    not_re_define: bool = field(default=False)  # 内部标志：是否禁止重新定义（你的工程自用）
    valid_on_test: bool = field(default=False)  # 是否在 test 集上做验证（有些竞赛/设置会用）
    node_level_task: bool = field(default=False)  # 是否为节点级任务（默认是图级任务）
    continuous_feature: bool = field(default=False)  # 输入特征是否为连续值（而非离散 id）
    multi_label: int = field(default=0)  # 多标签分类的标签数（0 表示非多标签任务）
    fusion: str = field(default='None')  # 特征融合方式占位（与你工程中的融合策略对应）
    reattention: bool = field(default=False)  # 是否启用“再注意力”机制（工程自定义开关）

    # ===== DDI 任务的专属字段 =====
    add_substructure: str = field(default='transform')  # 子结构添加方式：pre/transform 等（预处理或训练阶段生成）
    use_lmdb_cache: bool = field(default=True)  # 是否使用 LMDB 缓存数据
    extra_method: str = field(default='adj')  # 额外构图/特征方法（如使用邻接等）
    local_attention_on_substructures: bool = field(default=True)  # 是否在子结构范围内使用局部注意力

    debug_model: bool = field(default=False)  # 调试模式（可能启用额外打印/断言）

    multiprocessing: bool = field(default=False)  # 是否启用 Python 多进程预处理/加载
    num_processes: int = field(default=64)  # 多进程数量

    dataset: str = field(default='zinc')  # 数据集别名（内部数据模块使用）
    split: str = field(default='given')  # 数据划分方式：given/随机 等
    root_folder: str = field(default='./datasets')  # 原始数据根目录（与 data_dir 区分：data_dir 多为中间/缓存目录）

    id_type: str = field(default='cycle_graph')  # 子结构 id 类型：如环/路径/星状/k 近邻等
    induced: bool = field(default=True)  # 是否使用诱导子图
    edge_automorphism: str = field(default='induced')  # 边自同构处理方式

    ks: str = field(default='[8]')  # 各子结构的 k（如 k-hop/k-邻域），字符串以便直接从命令行传入列表样式
    id_scope: str = field(default='global')  # 子结构编号作用域：global/local

    custom_edge_list: str = field(default='none', metadata={"help": "custom_edge_list"})  # 自定义边列表文件/标记（none 表示关闭）

    directed: bool = field(default=False)  # 图是否有向
    directed_orbits: bool = field(default=False)  # 有向图时是否使用有向 orbit（子结构轨道）

    subsampling: bool = field(default=True)  # 是否对子结构进行子采样（加速/正则）
    sampling_mode: str = field(default='min_set_cover_random')  # 采样模式：random/shortest_path/min_set_cover 等
    not_only_unused_nodes: bool = field(default=False)  # 采样时是否不只选择“未使用节点”（工程自定义语义）

    sampling_redundancy: int = field(default=2)  # 采样冗余度（每处多采样几份以提高覆盖）
    sampling_stride: int = field(default=5)  # 采样步长（窗口/间隔）
    sampling_random_rate: float = field(default=0.1)  # 随机采样比例
    sampling_random_init: bool = field(default=True)  # 是否随机初始化采样器

    must_select_sub: str = field(default='cycle_graph')  # 必选子结构类型（如必须包含环）
    lmdb_dir: str = field(default='cache')  # LMDB 相对目录
    lmdb_root_dir: str = field(default='./')  # LMDB 根目录
    recache_lmdb: bool = field(default=False)  # 是否强制重建 LMDB 缓存

    use_transform_cache: bool = field(default=False)  # 是否使用 transform 结果缓存
    transform_cache_number: int = field(default=100)  # transform 缓存条目上限
    recache_transform: bool = field(default=False)  # 是否强制重建 transform 缓存
    transform_dir: str = field(default='transform')  # transform 缓存目录

    fuse_method: str = field(default='concat',
                             metadata={"help": "Fusion method for dual graph encoding"})  # 双图编码融合方式（concat/mean/attn 等）
    do_project: bool = field(default=True,
                             metadata={"help": "Use projection head on graph representation"})  # 是否在图表示后接投影头

    # 这里覆盖父类默认值，固定为二分类（若你需要多分类/回归，可在命令行改回去）
    num_classes: int = field(default=1, metadata={
        "help": "Number of classes for classification or regression task"})  # 类别数（DDI常为二分类）

    max_position: int = field(default=100000,
                              metadata={"help": "Maximum number of nodes in a graph"})  # 最大位置编码长度（用于某些位置/索引上限）
    test_after_valid: bool = field(
        default=True,
        metadata={"help": "每次验证后顺带评估 test 集合（仅记录，不影响 early stop / 最佳指标 / 学习率）"}
    )
    output_folder:str = field(default='/root/autodl-tmp/zcj/deepgraph/result/')


@register_task("graph_prediction_ddi", dataclass=GraphPredictionDDIConfig)
class GraphPredictionDDITask(FairseqTask):
    """
    Graph prediction DDI (classification or regression) task.
    """


    @staticmethod
    def add_args(parser):
        print("是否add_args")
        # 1) 把 DDI 的 dataclass 字段注册成命令行参数
        task_group = parser.add_argument_group("task")
        gen_parser_from_dataclass(task_group, GraphPredictionDDIConfig())
        # 2) 兼容你旧命令的旗标名
        g = parser.add_argument_group("graph_prediction_ddi (aliases)")
        # g.add_argument("--dataset-name", dest="dataset_name", type=str)
        # g.add_argument("--dataset-source", dest="dataset_source", type=str)
        # g.add_argument("--data-dir", dest="data_dir", type=str)
        # g.add_argument("--user-data-dir", dest="user_data_dir", type=str, default="dataset")

    def __init__(self, cfg):
        super().__init__(cfg)


        if cfg.user_data_dir != "":
            print("在1错了")
            self.__import_user_defined_datasets(cfg.user_data_dir)
            print("在2错了")
            if cfg.dataset_name in DATASET_REGISTRY:
                print("在3错了")
                dataset_dict = DATASET_REGISTRY[cfg.dataset_name]
                self.dm = GraphormerDDIDataset(
                    dataset=dataset_dict["dataset"],
                    dataset_source=dataset_dict["source"],
                    train_idx=dataset_dict["train_idx"],
                    valid_idx=dataset_dict["valid_idx"],
                    test_idx=dataset_dict["test_idx"],
                    seed=cfg.seed)
            else:
                print("在4错了")
                raise ValueError(f"dataset {cfg.dataset_name} is not found in customized dataset module {cfg.user_data_dir}")
        else:

            kwargs={}
            kwargs['data_dir']=cfg.data_dir
            self.dm = GraphormerDDIDataset(
                dataset_spec=cfg.dataset_name,
                dataset_source=cfg.dataset_source,
                seed=cfg.seed,
                args=cfg,
                **kwargs
            )


    def __import_user_defined_datasets(self, dataset_dir):
        dataset_dir = dataset_dir.strip("/")
        module_parent, module_name = os.path.split(dataset_dir)
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
        for file in os.listdir(dataset_dir):
            path = os.path.join(dataset_dir, file)
            if (
                not file.startswith("_")
                and not file.startswith(".")
                and (file.endswith(".py") or os.path.isdir(path))
            ):
                task_name = file[: file.find(".py")] if file.endswith(".py") else file
                importlib.import_module(module_name + "." + task_name)

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        assert cfg.num_classes > 0, "Must set task.num_classes"
        return cls(cfg)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        assert split in ["train", "valid", "test","inner"]

        if split == "train":
            batched_data = self.dm.dataset_train
        elif split == "valid":
            batched_data = self.dm.dataset_val
        elif split == "test":
            batched_data = self.dm.dataset_test
        elif split == "inner":
            batched_data = self.dm.dataset


        len1=len(batched_data)
        batched_data = BatchedDataDataset_DDISubstructure(
            batched_data,
            max_node=self.max_nodes(),
            multi_hop_max_dist=self.cfg.multi_hop_max_dist,
            spatial_pos_max=self.cfg.spatial_pos_max,
        )
        len2=len(batched_data)
        print(f"{split},之前的batched_data的长度：{len1},之后的batched_data的长度：{len2}")
        if(len1!=len2):
            print("减少了")
        data_sizes = np.array([self.max_nodes()] * len(batched_data))

        target = TargetDataset(batched_data,self.cfg.node_level_task,max_node=self.max_nodes())
        #target = TargetDataset(batched_data, self.cfg.node_level_task)

        dataset = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": target,
            },
            sizes=data_sizes,
        )

        if split == "train" and self.cfg.train_epoch_shuffle:
            dataset = EpochShuffleDataset(
                dataset, size=len(dataset), seed=self.cfg.seed
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, cfg):
        from fairseq import models

        with open_dict(cfg) if OmegaConf.is_config(cfg) else contextlib.ExitStack():
            cfg.max_nodes = self.cfg.max_nodes

        cfg.subgraph_max_size = self.cfg.subgraph_max_size
        model = models.build_model(cfg, self)

        return model

    def max_nodes(self):
        return self.cfg.max_nodes

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        return None

    @property
    def label_dictionary(self):
        return None


    def valid_step(self, sample, model, criterion):

        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)


        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        """Aggregate logging outputs from data parallel training."""
        # backward compatibility for tasks that override aggregate_logging_outputs
        base_func = FairseqTask.aggregate_logging_outputs
        self_func = getattr(self, "aggregate_logging_outputs").__func__
        if self_func is not base_func:
            utils.deprecation_warning(
                "Tasks should implement the reduce_metrics API. "
                "Falling back to deprecated aggregate_logging_outputs API."
            )
            agg_logging_outputs = self.aggregate_logging_outputs(
                logging_outputs, criterion
            )
            for k, v in agg_logging_outputs.items():
                metrics.log_scalar(k, v)
            return

        if not any("ntokens" in log for log in logging_outputs):
            warnings.warn(
                "ntokens not found in Criterion logging outputs, cannot log wpb or wps"
            )
        else:
            ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
            metrics.log_scalar("wpb", ntokens, priority=180, round=1)
            metrics.log_speed("wps", ntokens, priority=90, round=1)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
            metrics.log_scalar("bsz", nsentences, priority=190, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)



