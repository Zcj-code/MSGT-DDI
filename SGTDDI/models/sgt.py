# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture, BaseFairseqModel,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from ..modules import init_graphormer_params, SGTGraphEncoder

logger = logging.getLogger(__name__)

from ..pretrain import load_pretrained_model



# @register_model("deepgraph")
@register_model("sgtddi")
# class DeepGraphModel(FairseqEncoderModel):
class SGTModel(FairseqEncoderModel):
    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        if getattr(args, "apply_graphormer_init", False):
            self.apply(init_graphormer_params)
        self.encoder_embed_dim = args.encoder_embed_dim
        self.num_classes = args.num_classes
        self.max_positions_value = args.max_position

        if args.pretrained_model_name != "none":
            self.load_state_dict(load_pretrained_model(args.pretrained_model_name))
            if not args.load_pretrained_model_output_layer:
                self.encoder.reset_output_layer_parameters()


    def max_positions(self):
        return self.max_positions_value  # 返回存储的最大节点数
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embeddings",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        # Arguments related to parameter initialization
        parser.add_argument(
            "--apply-graphormer-init",
            action="store_true",
            help="use custom param initialization for Graphormer",
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--pre-layernorm",
            action="store_true",
            help="apply layernorm before self-attention and ffn. Without this, post layernorm will used",
        )

        parser.add_argument("--deepnorm",action="store_true",)

        parser.add_argument("--deepnorm_encoder_only",action="store_true",)

        parser.add_argument("--deepnorm_decoder_only",action="store_true",)

        parser.add_argument("--do_project",action="store_true",)

        parser.add_argument(
            "--drug-feat-csv",
            type=str,  # 指定这个参数的类型为字符串
            default="/root/",  # 设置默认值
            help="Path to the drug feature CSV file"
        )


    def max_nodes(self):
        return self.encoder.max_nodes

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = SGTEncoder(args)
        base_model = cls(args, encoder)

        # === 包装为图对模型 ===
        # 默认使用 concat 策略融合两个图的表示
        model = SGTDDIWrapper(base_model, fuse_method=getattr(args, "fuse_method", "concat"),drug_feat_csv=args.drug_feat_csv)

        return model
        # return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)


class SGTEncoder(FairseqEncoder):
    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_nodes = args.max_nodes

        self.graph_encoder = SGTGraphEncoder(
            # < for graphormer
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            # >
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            pre_layernorm=args.pre_layernorm,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
            deepnorm=args.deepnorm,
            encoder_layers=args.encoder_layers,
            encode_adj = (hasattr(args,'extra_method') and args.extra_method == 'adj'),
            subgraph_max_size=args.subgraph_max_size if hasattr(args,'extra_method') else None,
            reattention=args.reattention,
            feature_dim = 64

        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.lm_output_learned_bias = None
        self.projection_head=None
        self.multi_label=args.multi_label
        self.num_classes=args.num_classes

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, "remove_head", False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                if self.multi_label >1:
                    self.embed_out = nn.Linear(
                        args.encoder_embed_dim, args.num_classes*self.multi_label, bias=False
                    )
                else:
                    self.embed_out = nn.Linear(
                        args.encoder_embed_dim, args.num_classes, bias=False
                    )
            else:
                raise NotImplementedError
        self.do_project=args.do_project
        if self.do_project:
            self.projection_head = nn.Sequential(nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim))

        self.fusion=args.fusion if 'fusion' in args else 'None' # fusion baseline
        if self.fusion=='gate':
            self.output_gate = nn.Linear(args.encoder_ffn_embed_dim, 1, bias=False)

        self.reattention=args.reattention if 'reattention' in args else False

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None,interstate=False, **unused):
        # print(batched_data['x'].shape)
        inner_states, graph_rep,att = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )
        if interstate:
            return inner_states,batched_data,att
        else:
            del att

        # apply fusion
        if self.fusion == 'gate':
            x = torch.cat([tem.unsqueeze(2) for tem in inner_states],2).transpose(0, 1)
            score=torch.softmax(self.output_gate(x).squeeze(3),dim=-1).unsqueeze(-1)
            x=(score*x).sum(-2)

        else:

            x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        if self.do_project:

            x= self.projection_head(graph_rep)

            return x
        else:
            x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

            # project back to size of vocabulary
            if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
            ):
                x = F.linear(x, self.graph_encoder.embed_tokens.weight)
            elif self.embed_out is not None:
                #only CLS result
                if self.multi_label > 1:
                    x = self.embed_out(x[:,0,:])
                    x=x.reshape([-1,self.multi_label,self.num_classes])
                else:
                    x = self.embed_out(x)
            if self.lm_output_learned_bias is not None:
                x = x + self.lm_output_learned_bias

            return x

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes

    def upgrade_state_dict_named(self, state_dict, name):
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if "embed_out.weight" in k or "lm_output_learned_bias" in k:
                    del state_dict[k]
        return state_dict



class SGTDDIWrapper(BaseFairseqModel):
    def __init__(self, base_model: SGTModel, fuse_method: str = "concat", drug_feat_csv= ""):
        super().__init__()
        self.base_model = base_model
        self.fuse_method = fuse_method.lower()
        embed_dim = base_model.encoder_embed_dim
        self.drug_feature_dict = {}
        self.external_feature_dim = 0  # 用于后面拼接计算

        # ==== 读取 CSV 药物特征 ====
        drug_feat_csv=drug_feat_csv
        drug_feat_csv=None
        if drug_feat_csv is not None:
            df = pd.read_csv(drug_feat_csv, index_col="ID")
            self.drug_feature_dict = {
                drug_id: torch.tensor(row.values, dtype=torch.float32)
                for drug_id, row in df.iterrows()
            }
            self.external_feature_dim = df.shape[1]  # 特征维度

        if self.fuse_method == "concat":
            hidden_dim = embed_dim * 2 + self.external_feature_dim * 2
        elif self.fuse_method == "add":
            hidden_dim = embed_dim + self.external_feature_dim
        else:
            raise ValueError(f"Unsupported fuse_method: {fuse_method}")
        #
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim, hidden_dim // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(hidden_dim // 2, base_model.num_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 8, base_model.num_classes)
        )
        # print(hidden_dim)
        # self.classifier = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim // 2, base_model.num_classes)
        # )
        #self.max_positions = base_model.max_positions()

    def forward(self, batched_data, return_representation=False):
        # 拆分出两个图数据
        data1 = {k.replace("1", ""): v for k, v in batched_data.items() if k.endswith("1")}
        data2 = {k.replace("2", ""): v for k, v in batched_data.items() if k.endswith("2")}

        data1["idx"] = batched_data["idx"]
        data2["idx"] = batched_data["idx"]
        data1["global_idx"] = batched_data["global_idx1"]
        data2["global_idx"] = batched_data["global_idx2"]

        # 分别得到两个图的表示
        z1 = self.base_model(data1)  # shape: [B, D]
        z2 = self.base_model(data2)
        #print(f"z1 shape: {z1.shape}, z2 shape: {z2.shape}")  # 调试

        # 拼接外部药物特征
        if self.drug_feature_dict:
            features1 = [self.drug_feature_dict.get(drug_id, torch.zeros(self.external_feature_dim)) for drug_id in
                         data1["global_idx"]]
            features2 = [self.drug_feature_dict.get(drug_id, torch.zeros(self.external_feature_dim)) for drug_id in
                         data2["global_idx"]]


            features1 = torch.stack(features1).to(z1.device).to(z1.dtype)
            features2 = torch.stack(features2).to(z2.device).to(z2.dtype)

            z1 = torch.cat([z1, features1], dim=-1)
            z2 = torch.cat([z2, features2], dim=-1)


        # 融合图表示
        if self.fuse_method == "concat":
            z = torch.cat([z1, z2], dim=-1)
        elif self.fuse_method == "add":
            z = z1 + z2
        if return_representation:
            return z  # 返回融合后的表示
        # DDI 分类/回归输出
        logits = self.classifier(z)
        #print(f"logits (classifier output) shape: {logits.shape}")  # 调试

        return logits


    def set_num_updates(self, num_updates):
        """支持Fairseq训练进度更新"""
        super().set_num_updates(num_updates)
        if hasattr(self.base_model, 'set_num_updates'):
            self.base_model.set_num_updates(num_updates)
    def max_positions(self):
        # 通过 base_model 获取 max_positions
        return self.base_model.max_positions()  # 调用 base_model 中的方法
#

def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.deepnorm = getattr(args, "deepnorm", False)
    args.encoder_layers = getattr(args,"encoder_layers",12)
    args.do_project = getattr(args, "do_project",False)


@register_model_architecture("sgtddi", "sgt_ddi")
def graphormer_ddi_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 80)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 80)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", False)
    args.share_encoder_input_output_embed = getattr(
            args, "share_encoder_input_output_embed", False
        )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.pre_layernorm = getattr(args, "pre_layernorm", False)
    base_architecture(args)


