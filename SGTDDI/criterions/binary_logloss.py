# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os

from fairseq.dataclass.configs import FairseqDataclass

import torch
from torch.nn import functional
from fairseq.logging import  metrics
from fairseq import  utils
# from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import (
    roc_auc_score,  # AUC计算
    accuracy_score,  # 准确率
    precision_score,  # 精确率
    recall_score,  # 召回率
    f1_score, average_precision_score  # F1分数
)
from sklearn.metrics import roc_curve, auc, precision_recall_curve



@register_criterion("binary_logloss", dataclass=FairseqDataclass)
class GraphPredictionBinaryLogLoss(FairseqCriterion):
    """
    Implementation for the binary log loss used in graphormer model training.
    """

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        """计算给定样本的损失

              参数:
                  model: 要训练的模型 (DeepGraphDDIWrapper)
                  sample: 包含输入数据和标签的字典
                  reduce: 是否对损失进行求和归约

              返回:
                  tuple: (loss, sample_size, logging_output)
        """
        # 1. 获取基础信息
        batch_size = sample["nsamples"]  # 当前batch的样本数

        # 2. 获取模型输出
        # 从模型获取logits（模型最后会通过classifier输出）
        logits = model(**sample["net_input"])  # shape: [batch_size,  num_classes]


        # 3. 处理预测结果

        targets = model.get_targets(sample, [logits])  # 获取真实标签 [batch_size]

        # # 调试打印
        # print("\n=== 样本ID和标签 ===")
        # target = targets
        # sample_ids = sample['net_input']['batched_data']['idx'].cpu().numpy()
        # sample_idsx1 = sample['net_input']['batched_data']['global_idx1']
        # sample_idsx2 = sample['net_input']['batched_data']['global_idx2']
        # y=sample['net_input']['batched_data']['y']

        #
        # for i in range(len(sample_ids)):
        #     print(
        #         f"样本 {i}: ID={sample_ids[i]}, "
        #         f"x1样本 {i}: ID={sample_idsx1[i]}, "
        #         f"x2样本 {i}: ID={sample_idsx2[i]}, "
        #         f"真实标签={target[i].item():.2f}, "
        #         f"预测值={logits[i].sigmoid().item():.2f}"
        #         f"y真实={y[i]}"
        #     )
        # target = target.float().view(-1)
        # y_tensor = sample['net_input']['batched_data']['y'].to(target.device).float().view(-1)
        #
        # if not torch.allclose(target, y_tensor, atol=1e-5):
        #     print("❌ target 与 sample['y'] 不一致！详细如下：")
        #     for i in range(len(target)):
        #         print(f"[{i}] target: {target[i]:.4f}, y: {y_tensor[i]:.4f}")
        #     raise AssertionError("targets 与 sample['y'] 不一致！")
        #
        #
        # # if "y" in sample['net_input']['batched_data']:
        # #     if not torch.allclose(targets.float(), y_tensor, atol=1e-4):
        # #         print("\n❌ targets 与 sample['y'] 不一致！详细如下：")
        # #         for i in range(len(targets)):
        # #             print(f"[{i}] target: {targets[i].item():.4f}, y: {y_tensor[i].item():.4f}")
        # #         raise AssertionError("targets 与 sample['target'] 不一致！")
        # #assert torch.allclose(targets.float(), y_tensor, atol=1e-5), "targets与sample['target']不一致！"
        # # 检查形状
        # if logits.shape[0] != targets.shape[0]:
        #     print(f"❌ 错误：预测样本数={logits.shape[0]}, 真实标签数={targets.shape[0]}")
        # 获取药物对的 ID
        drug_a = sample['net_input']['batched_data']['global_idx1']  # 药物 A 的 ID
        drug_b = sample['net_input']['batched_data']['global_idx2']  # 药物 B 的 ID

        if(logits.shape!=targets.shape):
            # 调试打印
            print("\n=== 样本ID和标签 ===")
            target = targets
            sample_ids = sample['net_input']['batched_data']['idx'].cpu().numpy()
            sample_idsx1 = sample['net_input']['batched_data']['global_idx1']
            sample_idsx2 = sample['net_input']['batched_data']['global_idx2']
            y=sample['net_input']['batched_data']['y']


            for i in range(len(sample_ids)):
                print(
                    f"样本 {i}: ID={sample_ids[i]}, "
                    f"x1样本 {i}: ID={sample_idsx1[i]}, "
                    f"x2样本 {i}: ID={sample_idsx2[i]}, "
                    f"真实标签={target[i].item():.2f}, "
                    f"预测值={logits[i].sigmoid().item():.2f}"
                    f"y真实={y[i]}"
                )
            print("两个形状不相等")
            print(f"logits shape: {logits.shape}, targets shape: {targets.shape}")

        # print(f"输入样本数: {sample['nsamples']}")
        # print(f"net_input 样本数: {sample['net_input']['batched_data']['idx'].shape[0]}")
        # print(f"targets 样本数: {sample['target'].shape[0]}")

        # 4. 计算预测结果（用于准确率计算）
        probs = torch.sigmoid(logits)  # 计算概率 [batch_size, num_classes]
        preds = (probs > 0.5).long()  # 阈值0.5得到预测类别

        # 5. 计算损失
        logits_flatten = logits.reshape(-1)  # 展平 [batch_size * num_classes]
        targets_flatten = targets[:logits.size(0)].reshape(-1)  # 对齐形状

        # 创建有效值掩码（忽略NaN标签）
        mask = ~torch.isnan(targets_flatten)

        # 二元交叉熵损失
        loss = functional.binary_cross_entropy_with_logits(
            logits_flatten[mask].float(),  # 预测值
            targets_flatten[mask].float(),  # 真实值
            reduction="sum" if reduce else "none"
        )

        # L2 正则化
        l2_lambda = 1e-5  # 可根据模型大小调节
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
        loss = loss + l2_lambda * l2_norm

        # 6. 计算评价指标（只在训练或验证时计算）
        if model.training or not model.training:  # 同时适用于训练和验证
            # 转换为numpy用于计算指标
            probs_np = probs.detach().cpu().numpy()
            targets_np = targets.cpu().numpy()
            preds_np = preds.cpu().numpy()

            # 初始化指标字典
            metrics_dict = {}

            # 基础分类指标
            if len(np.unique(targets_np)) > 1:  # 确保有正负样本
                try:
                    metrics_dict['auc'] = roc_auc_score(targets_np, probs_np)
                except ValueError:
                    metrics_dict['auc'] = float('nan')

            # 其他指标（需要处理可能的NaN）
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics_dict.update({
                    'accuracy': accuracy_score(targets_np, preds_np),
                    'precision': precision_score(targets_np, preds_np, zero_division=0),
                    'recall': recall_score(targets_np, preds_np, zero_division=0),
                    'f1': f1_score(targets_np, preds_np, zero_division=0),
                    'AUPR': average_precision_score(targets_np, probs_np)
                })

        # 6. 准备日志输出
        logging_output = {
            "loss": loss.data,  # 损失值
            "sample_size": torch.sum(mask.type(torch.int64)),  # 有效样本数
            "nsentences": batch_size,  # 总样本数
            #"ntokens": sample["net_input"]["batched_data"].shape[1],  # 原子/节点数(有问题)
            "ncorrect": (preds == targets[:preds.size(0)]).sum(),  # 正确预测数
            "drug_a": drug_a,  # 药物 A 的 ID
            "drug_b": drug_b,  # 药物 B 的 ID
            "y_true": targets.cpu().numpy(),  # 真实标签
            "y_pred": preds.cpu().numpy(),  # 预测标签
            "y_score": probs.detach().cpu().numpy()  # 预测得分
        }
        logging_output.update(metrics_dict)

        return loss, batch_size, logging_output

        # sample_size = sample["nsamples"]
        #
        # with torch.no_grad():
        #     natoms = sample["net_input"]["batched_data"]["x"].shape[1]
        #
        # logits = model(**sample["net_input"])
        # logits = logits[:, 0, :]
        # targets = model.get_targets(sample, [logits])
        # preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
        #
        # logits_flatten = logits.reshape(-1)
        # targets_flatten = targets[: logits.size(0)].reshape(-1)
        # mask = ~torch.isnan(targets_flatten)
        # loss = functional.binary_cross_entropy_with_logits(
        #     logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
        # )
        #
        # logging_output = {
        #     "loss": loss.data,
        #     "sample_size": torch.sum(mask.type(torch.int64)),
        #     "nsentences": sample_size,
        #     "ntokens": natoms,
        #     "ncorrect": (preds == targets[:preds.size(0)]).sum(),
        # }
        # return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / sample_size, sample_size, round=1
            )
            # 其他指标
        for key in ["auc", "precision", "recall", "f1"]:
            values = [log.get(key, None) for log in logging_outputs if key in log]
            values = [v for v in values if v is not None and not isinstance(v, str)]
            if values:
                avg = sum(values) / len(values)
                metrics.log_scalar(key, avg, len(values), round=3)
            # AUPR
        for key in ["AUPR"]:
            values = [log.get(key, None) for log in logging_outputs if key in log]
            values = [v for v in values if v is not None and not isinstance(v, str)]
            if values:
                avg = sum(values) / len(values)
                metrics.log_scalar(key, avg, len(values), round=3)

            # Record Drug IDs and other info


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

# @register_criterion("binary_logloss_with_flag", dataclass=FairseqDataclass)
# class GraphPredictionBinaryLogLossWithFlag(GraphPredictionBinaryLogLoss):
#     """
#     Implementation for the binary log loss used in graphormer model training.
#     """
#
#     def forward(self, model, sample, reduce=True):
#         """Compute the loss for the given sample.
#
#         Returns a tuple with three elements:
#         1) the loss
#         2) the sample size, which is used as the denominator for the gradient
#         3) logging outputs to display while training
#         """
#         sample_size = sample["nsamples"]
#         perturb = sample.get("perturb", None)
#
#         batch_data = sample["net_input"]["batched_data"]["x"]
#         with torch.no_grad():
#             natoms = batch_data.shape[1]
#         logits = model(**sample["net_input"], perturb=perturb)[:, 0, :]
#         targets = model.get_targets(sample, [logits])
#         preds = torch.where(torch.sigmoid(logits) < 0.5, 0, 1)
#
#         logits_flatten = logits.reshape(-1)
#         targets_flatten = targets[: logits.size(0)].reshape(-1)
#         mask = ~torch.isnan(targets_flatten)
#         loss = functional.binary_cross_entropy_with_logits(
#             logits_flatten[mask].float(), targets_flatten[mask].float(), reduction="sum"
#         )
#
#         logging_output = {
#             "loss": loss.data,
#             "sample_size": logits.size(0),
#             "nsentences": sample_size,
#             "ntokens": natoms,
#             "ncorrect": (preds == targets[:preds.size(0)]).sum(),
#         }
#         return loss, sample_size, logging_output
