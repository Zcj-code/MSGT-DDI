"""Paper-oriented SGT-DDI case-study visualization (v2).

Drop-in replacement for ``SGTDDI/evaluate/explain_case_study.py``.

Why this version exists
-----------------------
The SGT-DDI data transform uses stochastic substructure subsampling
(`min_set_cover_random`) during ``dataset[idx]``. Therefore a single-sample
explanation can legitimately produce a probability different from the score
saved during the full test-set evaluation, even with the same checkpoint.

This script handles that explicitly:

1. Runs the target pair repeatedly with deterministic resampling seeds.
2. Reports MC mean/std/range and compares them with the saved test CSV score.
3. Aggregates atom attribution across resamples (atom indices are stable).
4. Aggregates substructures by (type, member atoms) and reports appearance rate.
5. Ranks stable substructures and optionally removes highly redundant overlaps.
6. Verifies RDKit-to-graph atom indexing using edge_index or spatial_pos==1 topology.
7. Keeps the explicit padding-mask diagnostic.
8. Uses RDKit Cairo for authoritative per-atom heatmap colors and produces
   substructure panels without a misleading 0-1 colorbar.

Important scientific interpretation
-----------------------------------
Attention/attention-rollout is an attribution signal, not a causal effect and
has no positive/negative direction. Use wording such as "highlighted",
"attended", or "associated with the model decision", rather than claiming that
an attended atom/substructure causes a DDI.
"""

from __future__ import annotations

import csv
import io
import json
import os
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Import RDKit depiction before graph_tool. This avoids the Boost.Python
# rdDepictor conflict seen in some environments.
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

import graph_tool  # noqa: F401
import joblib  # noqa: F401
import numpy as np
import torch
import torch_geometric  # noqa: F401
from fairseq import options, tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf


# =============================================================================
# Generic helpers
# =============================================================================


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(value: Any, device: torch.device, use_fp16: bool = False) -> Any:
    if torch.is_tensor(value):
        value = value.to(device)
        if use_fp16 and value.is_floating_point():
            value = value.half()
        return value
    if isinstance(value, dict):
        return {k: move_to_device(v, device, use_fp16) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device, use_fp16) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device, use_fp16) for v in value)
    return value


def tensor_to_python(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {k: tensor_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def first_item(value: Any) -> Any:
    while isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if torch.is_tensor(value):
        value = value.detach().cpu().reshape(-1)
        return value[0].item() if value.numel() else None
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0].item() if value.size else None
    return value


def select_batch_value(value: Any, batch_index: int) -> Any:
    """Select one sample from a tensor/list/tuple batch container."""
    if torch.is_tensor(value):
        return value[batch_index]
    if isinstance(value, (list, tuple)):
        return value[batch_index]
    if isinstance(value, np.ndarray):
        return value[batch_index]
    # Scalars are accepted only for a single-item batch.
    if batch_index != 0:
        raise IndexError(f"Cannot select batch index {batch_index} from scalar {type(value)}")
    return value


def normalize_drug_id(value: Any) -> str:
    value = first_item(value)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value).strip().strip("'\"")


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_scores(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(map(int, a)), set(map(int, b))
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


# =============================================================================
# Attention processing
# =============================================================================


def average_attention_heads(attn: torch.Tensor, batch_index: int = 0) -> torch.Tensor:
    """Return one [T,T] matrix for the requested batch item."""
    attn = attn.detach().float().cpu()

    if attn.dim() == 4:
        # SGT explain mode: [heads, batch, tokens, tokens]
        if batch_index >= attn.size(1):
            raise IndexError(
                f"batch_index={batch_index} outside attention batch dimension {attn.size(1)}"
            )
        return attn[:, batch_index].mean(dim=0)

    if attn.dim() == 3:
        # In this project explain attention is normally 4-D. Handle common
        # fallbacks conservatively.
        if attn.size(0) == 1:
            if batch_index != 0:
                raise IndexError("3-D attention contains only one batch item")
            return attn[0]
        if batch_index == 0:
            return attn.mean(dim=0)  # assume [heads,T,T]
        raise ValueError(
            "Ambiguous 3-D attention for batch_index>0; expected 4-D [H,B,T,T]."
        )

    if attn.dim() == 2:
        if batch_index != 0:
            raise IndexError("2-D attention contains only one sample")
        return attn

    raise ValueError(f"Unsupported attention shape: {tuple(attn.shape)}")


def validate_attention_size(attn: torch.Tensor, required_size: int, layer_index: int) -> None:
    if attn.size(-1) < required_size or attn.size(-2) < required_size:
        raise ValueError(
            f"Layer {layer_index} attention {tuple(attn.shape)} is smaller than "
            f"required ({required_size},{required_size})."
        )


def attention_to_node_scores(
    attentions: Sequence[Optional[torch.Tensor]],
    node_count: int,
    method: str,
    batch_index: int,
) -> Dict[str, np.ndarray]:
    valid = [a for a in attentions if a is not None]
    if not valid:
        raise ValueError("No attention tensors were returned by the model.")

    size = node_count + 1  # CLS + nodes

    if method == "last_cls":
        last = average_attention_heads(valid[-1], batch_index=batch_index)
        validate_attention_size(last, size, len(valid) - 1)
        last = torch.nan_to_num(last[:size, :size], nan=0.0, posinf=0.0, neginf=0.0)
        raw = last[0, 1:size].numpy()
        return {"raw": raw, "normalized_all_nodes": normalize_scores(raw)}

    if method != "rollout":
        raise ValueError(f"Unsupported attention method: {method}")

    joint = torch.eye(size, dtype=torch.float32)
    for layer_index, attn in enumerate(valid):
        a = average_attention_heads(attn, batch_index=batch_index)
        validate_attention_size(a, size, layer_index)
        a = torch.nan_to_num(a[:size, :size], nan=0.0, posinf=0.0, neginf=0.0)
        # Residual-aware rollout.
        a = a + torch.eye(size, dtype=a.dtype)
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        joint = a.matmul(joint)

    raw = joint[0, 1:size].numpy()
    return {"raw": raw, "normalized_all_nodes": normalize_scores(raw)}


def last_layer_attention_diagnostics(
    attentions: Sequence[Optional[torch.Tensor]],
    node_count: int,
    batch_index: int,
) -> Dict[str, np.ndarray]:
    valid = [a for a in attentions if a is not None]
    if not valid:
        raise ValueError("No attention tensors were returned by the model.")
    size = node_count + 1
    last = average_attention_heads(valid[-1], batch_index=batch_index)
    validate_attention_size(last, size, len(valid) - 1)
    last = torch.nan_to_num(last[:size, :size], nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "cls_to_node": last[0, 1:size].numpy(),
        "incoming_to_node": last[:, 1:size].sum(dim=0).numpy(),
        "node_to_cls": last[1:size, 0].numpy(),
        "row_sum": last.sum(dim=-1).numpy(),
    }


# =============================================================================
# Substructure type decoding
# =============================================================================


def parse_ks(value: Any) -> List[int]:
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    text = str(value).strip().strip("[]")
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_type_label_map(id_type: str, ks: Any) -> Dict[int, str]:
    """Decode the third identifier row into readable structural labels.

    For the user's configuration:
      cycle_graph k=8 -> type 0..5 = Cycle-3 .. Cycle-8
      path_graph  k=4 -> type 6..10 = Path-4 .. Path-8
      star_graph  k=6 -> type 11..15 = Star with 2..6 leaves
      k_neighborhood k=2 -> type -1 = 2-hop neighborhood
    """
    types = str(id_type).split("+")
    k_values = parse_ks(ks)
    labels: Dict[int, str] = {}
    next_type = 0
    neighbor_labels: List[str] = []

    for pos, kind in enumerate(types):
        k = k_values[pos] if pos < len(k_values) else None
        if kind == "cycle_graph" and k is not None:
            for n in range(3, k + 1):
                labels[next_type] = f"Cycle-{n}"
                next_type += 1
        elif kind == "path_graph" and k is not None:
            for n in range(k, 9):
                labels[next_type] = f"Path-{n}"
                next_type += 1
        elif kind == "star_graph" and k is not None:
            for leaves in range(2, k + 1):
                labels[next_type] = f"Star-{leaves} leaves"
                next_type += 1
        elif kind == "complete_graph" and k is not None:
            for n in range(3, k + 1):
                labels[next_type] = f"Clique-{n}"
                next_type += 1
        elif kind == "k_neighborhood" and k is not None:
            neighbor_labels.append(f"{k}-hop neighborhood")
        elif kind == "random_walk" and k is not None:
            neighbor_labels.append(f"Random walk (length={k})")
        else:
            # Unknown predefined families still consume IDs in the transform,
            # but their exact count cannot be inferred safely here.
            # Do not fabricate a mapping.
            pass

    if len(neighbor_labels) == 1:
        labels[-1] = neighbor_labels[0]
    elif neighbor_labels:
        labels[-1] = " / ".join(neighbor_labels)
    else:
        labels[-1] = "Neighborhood-derived"
    return labels


def type_label(type_id: Optional[int], mapping: Dict[int, str]) -> str:
    if type_id is None:
        return "Unknown"
    return mapping.get(int(type_id), f"type_id={int(type_id)}")


# =============================================================================
# Molecule / substructure mapping and alignment
# =============================================================================


def unwrap_identifiers(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu().long()


def build_substructure_mapping(
    identifiers: Any,
    num_atoms: int,
    raw_node_scores: Sequence[float],
    atom_normalized_scores: Sequence[float],
    type_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    identifiers = unwrap_identifiers(identifiers)
    if identifiers is None or identifiers.numel() == 0:
        return []
    if identifiers.dim() != 2 or identifiers.size(0) < 2:
        raise ValueError(f"Unexpected identifiers shape: {tuple(identifiers.shape)}")

    raw_node_scores = np.asarray(raw_node_scores, dtype=float)
    atom_normalized_scores = np.asarray(atom_normalized_scores, dtype=float)

    sub_ids = identifiers[0].tolist()
    ordered: List[int] = []
    seen = set()
    for sid in sub_ids:
        sid = int(sid)
        if sid not in seen:
            seen.add(sid)
            ordered.append(sid)

    result: List[Dict[str, Any]] = []
    for order, sid in enumerate(ordered):
        mask = identifiers[0] == sid
        atoms = sorted(set(int(x) for x in identifiers[1, mask].tolist()))
        token_index = num_atoms + order
        tid: Optional[int] = None
        if identifiers.size(0) > 2:
            vals = identifiers[2, mask].tolist()
            if vals:
                tid = int(vals[0])

        raw_score = (
            float(raw_node_scores[token_index])
            if 0 <= token_index < len(raw_node_scores)
            else None
        )
        valid_atoms = [a for a in atoms if 0 <= a < len(atom_normalized_scores)]
        atom_support = (
            float(np.mean(atom_normalized_scores[valid_atoms])) if valid_atoms else None
        )
        result.append(
            {
                "sub_id": sid,
                "token_index": int(token_index),
                "atom_indices": atoms,
                "num_atoms": len(atoms),
                "type_id": tid,
                "type_label": type_label(tid, type_map),
                "raw_score": raw_score,
                "member_atom_mean_attribution": atom_support,
            }
        )

    # Normalize only among substructure tokens. This is more interpretable than
    # normalizing atoms and substructure tokens together, because token raw scores
    # are usually on a different scale.
    valid_scores = [s["raw_score"] for s in result if s["raw_score"] is not None]
    norm = normalize_scores(valid_scores)
    cursor = 0
    for item in result:
        if item["raw_score"] is not None:
            item["within_substructure_normalized_score"] = float(norm[cursor])
            cursor += 1
        else:
            item["within_substructure_normalized_score"] = None
    return result


def read_smiles_table(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    table: Dict[str, str] = {}
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return table
        lower = {x.lower(): x for x in reader.fieldnames}
        id_key = lower.get("id") or lower.get("drug_id") or lower.get("dbid")
        smiles_key = lower.get("smiles") or lower.get("canonical_smiles")
        if id_key is None or smiles_key is None:
            raise ValueError(
                f"SMILES CSV needs ID/drug_id and SMILES/canonical_smiles; got {reader.fieldnames}"
            )
        for row in reader:
            did = str(row.get(id_key, "")).strip().strip("'\"")
            smi = str(row.get(smiles_key, "")).strip()
            if did and smi:
                table[did] = smi
    return table


def atom_symbols(smiles: Optional[str]) -> List[str]:
    if not smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [a.GetSymbol() for a in mol.GetAtoms()]


def undirected_atom_edges(edge_index: Any, num_atoms: int) -> set:
    """Recover indexed undirected atom-atom edges from a PyG edge_index."""
    if edge_index is None:
        return set()
    if not torch.is_tensor(edge_index):
        try:
            edge_index = torch.as_tensor(edge_index)
        except Exception:
            return set()
    edge_index = edge_index.detach().cpu().long()
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        return set()
    edges = set()
    for u, v in edge_index.t().tolist():
        u, v = int(u), int(v)
        if u == v:
            continue
        if 0 <= u < num_atoms and 0 <= v < num_atoms:
            edges.add(tuple(sorted((u, v))))
    return edges


def atom_edges_from_spatial_pos(
    spatial_pos: Any,
    num_atoms: int,
    batch_index: int = 0,
) -> set:
    """Recover direct chemical bonds from Graphormer's shortest-path matrix.

    In ``preprocess_item_pair`` the shortest-path matrix is computed after token
    augmentation. A direct atom-atom chemical bond still has shortest-path distance
    exactly 1, whereas an atom pair connected only through a substructure token has
    distance >=2. Restricting to the first ``num_atoms`` nodes therefore provides a
    robust fallback when PyG ``edge_index`` is not exposed by the collated batch.
    """
    if spatial_pos is None:
        return set()
    if not torch.is_tensor(spatial_pos):
        try:
            spatial_pos = torch.as_tensor(spatial_pos)
        except Exception:
            return set()
    sp = spatial_pos.detach().cpu().long()
    if sp.dim() == 3:
        if not (0 <= batch_index < sp.size(0)):
            return set()
        sp = sp[batch_index]
    if sp.dim() != 2:
        return set()
    n = min(num_atoms, sp.size(0), sp.size(1))
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if int(sp[u, v].item()) == 1 or int(sp[v, u].item()) == 1:
                edges.add((u, v))
    return edges


def verify_atom_alignment(
    smiles: Optional[str],
    graph_atom_count: int,
    edge_index: Any = None,
    spatial_pos: Any = None,
    batch_index: int = 0,
) -> Dict[str, Any]:
    """Verify that RDKit atom indices coincide with graph atom indices.

    We first try the original/augmented PyG ``edge_index``. If it is unavailable or
    contains no recoverable atom-atom edges, we fall back to ``spatial_pos == 1`` in
    the real-atom block. Full equality of the indexed bond set is considerably stronger
    than the old atom-count-only check.
    """
    if not smiles:
        return {
            "status": "missing_smiles",
            "graph_num_atoms": graph_atom_count,
            "exact_order_verified": False,
            "edge_source": None,
        }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "status": "invalid_smiles",
            "graph_num_atoms": graph_atom_count,
            "exact_order_verified": False,
            "edge_source": None,
        }

    rdkit_edges = {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()
    }

    graph_edges_edge_index = undirected_atom_edges(edge_index, graph_atom_count)
    graph_edges_spatial = atom_edges_from_spatial_pos(
        spatial_pos, graph_atom_count, batch_index=batch_index
    )

    if graph_edges_edge_index:
        graph_edges = graph_edges_edge_index
        edge_source = "edge_index"
    elif graph_edges_spatial:
        graph_edges = graph_edges_spatial
        edge_source = "spatial_pos==1"
    else:
        graph_edges = set()
        edge_source = "unavailable"

    count_matches = mol.GetNumAtoms() == graph_atom_count
    topology_matches = count_matches and bool(graph_edges) and rdkit_edges == graph_edges

    rdkit_degree = [0] * mol.GetNumAtoms()
    for u, v in rdkit_edges:
        rdkit_degree[u] += 1
        rdkit_degree[v] += 1
    graph_degree = [0] * graph_atom_count
    for u, v in graph_edges:
        if u < graph_atom_count and v < graph_atom_count:
            graph_degree[u] += 1
            graph_degree[v] += 1
    degree_matches = count_matches and rdkit_degree == graph_degree

    missing_from_graph = sorted(rdkit_edges - graph_edges)
    extra_in_graph = sorted(graph_edges - rdkit_edges)
    return {
        "status": "ok" if count_matches else "atom_count_mismatch",
        "mol_num_atoms": mol.GetNumAtoms(),
        "graph_num_atoms": graph_atom_count,
        "count_matches": count_matches,
        "edge_source": edge_source,
        "rdkit_bond_count": len(rdkit_edges),
        "graph_atom_edge_count": len(graph_edges),
        "edge_index_atom_edge_count": len(graph_edges_edge_index),
        "spatial_pos_atom_edge_count": len(graph_edges_spatial),
        "indexed_degree_sequence_matches": degree_matches,
        "topology_matches_exact_indexing": topology_matches,
        "exact_order_verified": topology_matches,
        "missing_rdkit_edges_in_graph": missing_from_graph[:20],
        "extra_graph_edges": extra_in_graph[:20],
        "atom_symbols": [a.GetSymbol() for a in mol.GetAtoms()],
        "note": (
            "Exact indexing is accepted when atom count and the complete indexed "
            "undirected atom-bond set match RDKit. If edge_index is unavailable, "
            "direct bonds are reconstructed from spatial_pos==1 in the real-atom block."
        ),
    }


# =============================================================================
# Padding diagnostics
# =============================================================================


def inspect_padding_rule(
    x: torch.Tensor,
    actual_padding_mask: Optional[torch.Tensor],
    batch_index: int,
    smiles: Optional[str],
    num_atoms: int,
    last_diag: Dict[str, np.ndarray],
    zero_epsilon: float,
) -> Dict[str, Any]:
    x_cpu = x.detach().float().cpu()
    if x_cpu.dim() != 3:
        raise ValueError(f"Expected x [B,N,F], got {tuple(x_cpu.shape)}")

    first_feature = x_cpu[batch_index, :num_atoms, 0].numpy()
    legacy_mask = first_feature == 0

    if actual_padding_mask is None:
        full_mask = x_cpu.abs().sum(dim=-1).eq(0)
        source = "fallback all-zero node feature"
    else:
        full_mask = actual_padding_mask.detach().bool().cpu()
        source = "explicit collator padding_mask"
    actual_mask = full_mask[batch_index, :num_atoms].numpy().astype(bool)

    symbols = atom_symbols(smiles)
    if len(symbols) != num_atoms:
        symbols = ["?"] * num_atoms

    incoming = np.asarray(last_diag["incoming_to_node"][:num_atoms], dtype=float)
    cls = np.asarray(last_diag["cls_to_node"][:num_atoms], dtype=float)

    rows = []
    for i in range(num_atoms):
        rows.append(
            {
                "atom_index": i,
                "symbol": symbols[i],
                "first_feature": float(first_feature[i]),
                "actual_marked_as_padding": bool(actual_mask[i]),
                "legacy_would_mark_as_padding": bool(legacy_mask[i]),
                "last_layer_incoming_attention": float(incoming[i]),
                "last_layer_cls_attention": float(cls[i]),
                "incoming_is_near_zero": bool(abs(incoming[i]) <= zero_epsilon),
            }
        )

    actual_indices = np.flatnonzero(actual_mask).tolist()
    legacy_indices = np.flatnonzero(legacy_mask).tolist()
    non_carbon = [i for i, s in enumerate(symbols) if s not in ("C", "?")]
    return {
        "active_mask_source": source,
        "actual_masked_atom_indices": actual_indices,
        "num_actual_masked_atoms": len(actual_indices),
        "suspected_padding_mask_bug": bool(actual_indices),
        "legacy_rule_checked": "x[:, :, 0] == 0",
        "legacy_masked_atom_indices": legacy_indices,
        "non_carbon_atom_indices": non_carbon,
        "legacy_non_carbon_masked_atom_indices": [i for i in non_carbon if legacy_mask[i]],
        "all_non_carbon_atoms_would_be_masked_by_legacy_rule": bool(non_carbon)
        and all(legacy_mask[i] for i in non_carbon),
        "atom_rows": rows,
    }


def print_padding_diagnostic(drug_id: str, diagnostic: Dict[str, Any]) -> None:
    print(f"\n[{drug_id}] padding-mask diagnostic")
    print(f"  source: {diagnostic['active_mask_source']}")
    print(f"  actual masked real atoms: {diagnostic['actual_masked_atom_indices']}")
    print(f"  legacy x[...,0]==0 would mask: {diagnostic['legacy_masked_atom_indices']}")
    print(f"  active padding-mask bug: {diagnostic['suspected_padding_mask_bug']}")


# =============================================================================
# Expected test CSV
# =============================================================================


def _lower_key_map(fieldnames: Sequence[str]) -> Dict[str, str]:
    return {str(x).lower(): x for x in fieldnames}


def read_prediction_csv_row(
    path: Optional[str],
    requested_index: int,
    drug1: Optional[str] = None,
    drug2: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            return None
        keys = _lower_key_map(reader.fieldnames or [])
        a_key = keys.get("drug_a") or keys.get("drug1")
        b_key = keys.get("drug_b") or keys.get("drug2")
        score_key = keys.get("score") or keys.get("probability") or keys.get("prob")
        pred_key = keys.get("pred") or keys.get("prediction")
        label_key = keys.get("label") or keys.get("target")

        def convert(row: Dict[str, str], row_index: int) -> Dict[str, Any]:
            return {
                "csv_path": path,
                "row_index": row_index,
                "drug_a": row.get(a_key) if a_key else None,
                "drug_b": row.get(b_key) if b_key else None,
                "score": safe_float(row.get(score_key)) if score_key else None,
                "pred": int(float(row[pred_key])) if pred_key and row.get(pred_key) not in (None, "") else None,
                "label": int(float(row[label_key])) if label_key and row.get(label_key) not in (None, "") else None,
            }

        candidate = convert(rows[requested_index], requested_index) if 0 <= requested_index < len(rows) else None
        if candidate and drug1 and drug2:
            ca, cb = str(candidate.get("drug_a")), str(candidate.get("drug_b"))
            if (ca, cb) == (drug1, drug2) or (ca, cb) == (drug2, drug1):
                candidate["pair_match"] = True
                return candidate

            # Fall back to searching the pair in the CSV, in either orientation.
            for i, row in enumerate(rows):
                found = convert(row, i)
                fa, fb = str(found.get("drug_a")), str(found.get("drug_b"))
                if (fa, fb) == (drug1, drug2) or (fa, fb) == (drug2, drug1):
                    found["pair_match"] = True
                    found["requested_index_row_mismatch"] = True
                    return found
            candidate["pair_match"] = False
        return candidate


def completed_checkpoint_epoch(state: Dict[str, Any]) -> Optional[int]:
    """Convert Fairseq iterator state to the epoch whose evaluation just finished."""
    epoch = state.get("epoch") if isinstance(state, dict) else None
    it = state.get("iterations_in_epoch") if isinstance(state, dict) else None
    try:
        epoch_i = int(epoch)
    except (TypeError, ValueError):
        return None
    # Fairseq commonly stores epoch=E+1, iterations_in_epoch=0 immediately after E.
    try:
        if int(it) == 0 and epoch_i > 1:
            return epoch_i - 1
    except (TypeError, ValueError):
        pass
    return epoch_i


def _prediction_csv_candidates(
    args: Any, checkpoint_state: Dict[str, Any]
) -> List[Path]:
    """Find plausible saved test CSVs without scanning the whole filesystem."""
    roots: List[Path] = []
    explicit_dir = getattr(args, "prediction_csv_dir", None)
    if explicit_dir:
        roots.append(Path(explicit_dir))

    output_folder = getattr(args, "output_folder", None)
    if output_folder:
        roots.append(Path(output_folder))

    cwd = Path.cwd()
    roots.extend([cwd / "result", cwd / "results"])

    # De-duplicate existing roots.
    unique_roots: List[Path] = []
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key not in seen and root.exists():
            seen.add(key)
            unique_roots.append(root)

    epoch = completed_checkpoint_epoch(checkpoint_state)
    patterns = []
    if epoch is not None:
        patterns.extend([f"epoch_{epoch}_test_*.csv", f"epoch_{epoch:03d}_test_*.csv"])
    patterns.append("epoch_*_test_*.csv")

    found: List[Path] = []
    found_seen = set()
    for root in unique_roots:
        for pattern in patterns:
            for p in root.rglob(pattern):
                s = str(p.resolve())
                if s not in found_seen:
                    found_seen.add(s)
                    found.append(p)

    save_tag = Path(str(getattr(args, "save_dir", ""))).name
    tag_variants = [save_tag]
    if save_tag.startswith("ddi-"):
        tag_variants.append(save_tag[4:])

    def rank(p: Path) -> Tuple[int, int, float]:
        path_text = str(p)
        tag_hit = max((1 if t and t in path_text else 0) for t in tag_variants)
        epoch_hit = 0
        if epoch is not None and re.search(rf"epoch_0*{epoch}_test_", p.name):
            epoch_hit = 1
        try:
            mt = p.stat().st_mtime
        except OSError:
            mt = 0.0
        return tag_hit, epoch_hit, mt

    found.sort(key=rank, reverse=True)
    return found


def resolve_prediction_csv(
    args: Any,
    checkpoint_state: Dict[str, Any],
    requested_index: int,
    drug1: str,
    drug2: str,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """Resolve the saved test CSV and return both row and resolution diagnostics."""
    explicit = getattr(args, "prediction_csv", None)
    if explicit:
        row = read_prediction_csv_row(explicit, requested_index, drug1, drug2)
        return row, {
            "mode": "explicit",
            "selected_path": explicit,
            "candidate_count": 1,
            "completed_checkpoint_epoch": completed_checkpoint_epoch(checkpoint_state),
        }

    candidates = _prediction_csv_candidates(args, checkpoint_state)
    attempts = []
    best_row = None
    best_path = None
    for p in candidates:
        try:
            row = read_prediction_csv_row(str(p), requested_index, drug1, drug2)
        except Exception as exc:
            attempts.append({"path": str(p), "error": str(exc)})
            continue
        attempts.append(
            {
                "path": str(p),
                "pair_match": None if row is None else row.get("pair_match"),
                "score": None if row is None else row.get("score"),
            }
        )
        if row is not None and row.get("pair_match"):
            best_row = row
            best_path = str(p)
            break
        if best_row is None and row is not None:
            best_row, best_path = row, str(p)

    return best_row, {
        "mode": "auto" if candidates else "not_found",
        "selected_path": best_path,
        "candidate_count": len(candidates),
        "completed_checkpoint_epoch": completed_checkpoint_epoch(checkpoint_state),
        "attempts": attempts[:20],
    }


# =============================================================================
# Model / dataset execution
# =============================================================================


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: str, model_cfg: Any) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    try:
        model.load_state_dict(model_state, strict=True, model_cfg=model_cfg)
    except TypeError:
        model.load_state_dict(model_state, strict=True)


def checkpoint_training_state(checkpoint_path: str) -> Dict[str, Any]:
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
        extra = state.get("extra_state", {}) if isinstance(state, dict) else {}
        iterator = extra.get("train_iterator", {}) if isinstance(extra, dict) else {}
        opt_hist = state.get("optimizer_history", []) if isinstance(state, dict) else []
        last_opt = opt_hist[-1] if opt_hist else {}
        return {
            "epoch": iterator.get("epoch"),
            "iterations_in_epoch": iterator.get("iterations_in_epoch"),
            "num_updates": last_opt.get("num_updates"),
            "best": extra.get("best") if isinstance(extra, dict) else None,
        }
    except Exception as exc:
        return {"read_error": str(exc)}


def logits_max_abs_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float().cpu() - b.detach().float().cpu()).abs().max().item())


def natural_context_indices(target_index: int, dataset_len: int, batch_size: int) -> List[int]:
    if batch_size <= 1:
        return [target_index]
    start = (target_index // batch_size) * batch_size
    end = min(start + batch_size, dataset_len)
    return list(range(start, end))


def build_context_sample(dataset: Any, target_index: int, context_batch_size: int):
    indices = natural_context_indices(target_index, len(dataset), context_batch_size)
    items = [dataset[i] for i in indices]
    sample = dataset.collater(items)
    if not sample:
        raise RuntimeError("Dataset collater returned an empty sample.")

    batched = sample["net_input"]["batched_data"]
    batch_ids = batched.get("idx")
    if batch_ids is None:
        raise KeyError("batched_data does not contain idx")
    ids = batch_ids.detach().cpu().reshape(-1).tolist() if torch.is_tensor(batch_ids) else list(batch_ids)
    matches = [i for i, idx in enumerate(ids) if int(idx) == int(target_index)]
    if not matches:
        raise RuntimeError(
            f"Target index {target_index} was not retained by collator. Batch idx values: {ids}"
        )
    local_index = matches[0]
    target_item = items[indices.index(target_index)]
    return sample, target_item, local_index, indices


def prediction_summary(target: Any, probability: float, threshold: float) -> Dict[str, Any]:
    target_float = safe_float(first_item(target))
    pred = int(probability >= threshold)
    out = {"threshold": threshold, "predicted_label": pred, "target_label": None, "is_correct": None}
    if target_float is not None:
        label = int(target_float >= 0.5)
        out["target_label"] = label
        out["is_correct"] = pred == label
        out["confusion_type"] = {
            (1, 1): "true_positive",
            (0, 0): "true_negative",
            (1, 0): "false_positive",
            (0, 1): "false_negative",
        }[(pred, label)]
    return out


def run_once(
    *,
    model: torch.nn.Module,
    dataset: Any,
    target_index: int,
    context_batch_size: int,
    run_seed: int,
    device: torch.device,
    use_fp16: bool,
    attention_method: str,
    smoke_tolerance: float,
    strict_smoke: bool,
    smiles_table: Dict[str, str],
    type_map: Dict[int, str],
    zero_epsilon: float,
) -> Dict[str, Any]:
    # Dataset transform randomness is NumPy-based; reset BEFORE dataset[idx].
    set_all_seeds(run_seed)
    sample_cpu, target_item, local_index, context_indices = build_context_sample(
        dataset, target_index, context_batch_size
    )
    sample = move_to_device(sample_cpu, device, use_fp16=use_fp16)
    batched = sample["net_input"]["batched_data"]

    with torch.no_grad():
        normal_logits = model(batched)
        output = model(batched, return_explain=True)

    smoke_diff = logits_max_abs_difference(normal_logits, output["logits"])
    if smoke_diff > smoke_tolerance:
        msg = f"Normal/explain logit diff {smoke_diff:.6g} > {smoke_tolerance:.6g}"
        if strict_smoke:
            raise RuntimeError(msg)
        warnings.warn(msg)

    meta = output["meta"]
    n_atoms1 = int(first_item(select_batch_value(meta["num_atoms1"], local_index)))
    n_atoms2 = int(first_item(select_batch_value(meta["num_atoms2"], local_index)))
    n_sub1 = int(first_item(select_batch_value(meta["num_subtokens1"], local_index)))
    n_sub2 = int(first_item(select_batch_value(meta["num_subtokens2"], local_index)))
    total1, total2 = n_atoms1 + n_sub1, n_atoms2 + n_sub2

    pack1 = attention_to_node_scores(output["att1"], total1, attention_method, local_index)
    pack2 = attention_to_node_scores(output["att2"], total2, attention_method, local_index)
    last1 = last_layer_attention_diagnostics(output["att1"], total1, local_index)
    last2 = last_layer_attention_diagnostics(output["att2"], total2, local_index)

    drug1 = normalize_drug_id(select_batch_value(meta["global_idx1"], local_index))
    drug2 = normalize_drug_id(select_batch_value(meta["global_idx2"], local_index))
    smiles1, smiles2 = smiles_table.get(drug1), smiles_table.get(drug2)

    raw1, raw2 = pack1["raw"], pack2["raw"]
    raw_atom1, raw_atom2 = raw1[:n_atoms1], raw2[:n_atoms2]
    # IMPORTANT: atom plots normalize among atoms only, not atoms+subtokens.
    norm_atom1, norm_atom2 = normalize_scores(raw_atom1), normalize_scores(raw_atom2)

    identifiers1 = select_batch_value(meta["identifiers1"], local_index)
    identifiers2 = select_batch_value(meta["identifiers2"], local_index)
    subs1 = build_substructure_mapping(identifiers1, n_atoms1, raw1, norm_atom1, type_map)
    subs2 = build_substructure_mapping(identifiers2, n_atoms2, raw2, norm_atom2, type_map)

    pad1 = inspect_padding_rule(
        batched["x1"], batched.get("padding_mask1"), local_index, smiles1, n_atoms1, last1, zero_epsilon
    )
    pad2 = inspect_padding_rule(
        batched["x2"], batched.get("padding_mask2"), local_index, smiles2, n_atoms2, last2, zero_epsilon
    )

    target_batch = sample.get("target")
    target_value = select_batch_value(target_batch, local_index) if target_batch is not None else None
    probability = float(output["prob"][local_index].detach().float().cpu().reshape(-1)[0].item())
    logit = float(output["logits"][local_index].detach().float().cpu().reshape(-1)[0].item())

    edge1 = getattr(target_item, "edge_index1", None)
    edge2 = getattr(target_item, "edge_index2", None)

    return {
        "run_seed": run_seed,
        "context_indices": context_indices,
        "local_batch_index": local_index,
        "drug1": drug1,
        "drug2": drug2,
        "smiles1": smiles1,
        "smiles2": smiles2,
        "target": tensor_to_python(target_value),
        "probability": probability,
        "logit": logit,
        "smoke_diff": smoke_diff,
        "num_atoms1": n_atoms1,
        "num_atoms2": n_atoms2,
        "num_subtokens1": n_sub1,
        "num_subtokens2": n_sub2,
        "raw_atom_scores1": np.asarray(raw_atom1, dtype=float),
        "raw_atom_scores2": np.asarray(raw_atom2, dtype=float),
        "norm_atom_scores1": np.asarray(norm_atom1, dtype=float),
        "norm_atom_scores2": np.asarray(norm_atom2, dtype=float),
        "last_diag1": last1,
        "last_diag2": last2,
        "substructures1": subs1,
        "substructures2": subs2,
        "padding_diag1": pad1,
        "padding_diag2": pad2,
        "alignment1": verify_atom_alignment(
            smiles1,
            n_atoms1,
            edge_index=edge1,
            spatial_pos=batched.get("spatial_pos1"),
            batch_index=local_index,
        ),
        "alignment2": verify_atom_alignment(
            smiles2,
            n_atoms2,
            edge_index=edge2,
            spatial_pos=batched.get("spatial_pos2"),
            batch_index=local_index,
        ),
        "attention1": output["att1"],
        "attention2": output["att2"],
        "attention_shape1": list(output["att1"][0].shape) if output["att1"] else None,
        "attention_shape2": list(output["att2"][0].shape) if output["att2"] else None,
    }


# =============================================================================
# MC aggregation
# =============================================================================


def aggregate_atoms(runs: Sequence[Dict[str, Any]], key: str) -> Dict[str, np.ndarray]:
    arrays = [np.asarray(r[key], dtype=float) for r in runs]
    lengths = {len(x) for x in arrays}
    if len(lengths) != 1:
        raise ValueError(f"Atom count changed across resamples for {key}: {sorted(lengths)}")
    stack = np.stack(arrays, axis=0)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    return {
        "mean": mean,
        "std": std,
        "plot_score": normalize_scores(mean),
        "min": stack.min(axis=0),
        "max": stack.max(axis=0),
    }


def aggregate_substructures(
    runs: Sequence[Dict[str, Any]],
    key: str,
) -> List[Dict[str, Any]]:
    records: Dict[Tuple[str, Tuple[int, ...]], Dict[str, Any]] = {}
    total_runs = len(runs)

    for run_idx, run in enumerate(runs):
        seen_this_run = set()
        for sub in run[key]:
            atoms = tuple(sorted(int(x) for x in sub["atom_indices"]))
            label = str(sub.get("type_label", "Unknown"))
            rec_key = (label, atoms)
            if rec_key not in records:
                records[rec_key] = {
                    "type_label": label,
                    "type_id": sub.get("type_id"),
                    "atom_indices": list(atoms),
                    "num_atoms": len(atoms),
                    "raw_scores": [],
                    "within_scores": [],
                    "atom_support_scores": [],
                    "run_indices": [],
                }
            rec = records[rec_key]
            raw = safe_float(sub.get("raw_score"))
            within = safe_float(sub.get("within_substructure_normalized_score"))
            atom_support = safe_float(sub.get("member_atom_mean_attribution"))
            if raw is not None:
                rec["raw_scores"].append(raw)
            if within is not None:
                rec["within_scores"].append(within)
            if atom_support is not None:
                rec["atom_support_scores"].append(atom_support)
            if rec_key not in seen_this_run:
                rec["run_indices"].append(run_idx)
                seen_this_run.add(rec_key)

    out: List[Dict[str, Any]] = []
    for rec in records.values():
        appearance_count = len(set(rec["run_indices"]))
        appearance_rate = appearance_count / total_runs if total_runs else 0.0
        mean_raw = float(np.mean(rec["raw_scores"])) if rec["raw_scores"] else 0.0
        mean_within = float(np.mean(rec["within_scores"])) if rec["within_scores"] else 0.0
        mean_atom_support = (
            float(np.mean(rec["atom_support_scores"])) if rec["atom_support_scores"] else 0.0
        )
        # Stable direct token attribution. Appearance rate prevents a one-off random
        # sampled subgraph from dominating the paper figure.
        stability_score = appearance_rate * mean_within
        out.append(
            {
                "type_label": rec["type_label"],
                "type_id": rec["type_id"],
                "atom_indices": rec["atom_indices"],
                "num_atoms": rec["num_atoms"],
                "appearance_count": appearance_count,
                "appearance_rate": appearance_rate,
                "mean_raw_score": mean_raw,
                "std_raw_score": float(np.std(rec["raw_scores"])) if rec["raw_scores"] else 0.0,
                "mean_within_substructure_score": mean_within,
                "mean_member_atom_attribution": mean_atom_support,
                "stability_score": stability_score,
            }
        )

    out.sort(
        key=lambda x: (
            float(x["stability_score"]),
            float(x["mean_raw_score"]),
            float(x["appearance_rate"]),
        ),
        reverse=True,
    )
    return out


def select_nonredundant_substructures(
    ranked: Sequence[Dict[str, Any]], top_k: int, max_jaccard: float
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for candidate in ranked:
        if all(
            jaccard(candidate["atom_indices"], chosen["atom_indices"]) <= max_jaccard
            for chosen in selected
        ):
            selected.append(candidate)
            if len(selected) >= top_k:
                break
    # If the threshold was too strict, fill remaining slots by rank so the panel
    # still contains top_k items. The JSON keeps the overlap information.
    if len(selected) < top_k:
        for candidate in ranked:
            if candidate not in selected:
                selected.append(candidate)
                if len(selected) >= top_k:
                    break
    return selected


def probability_diagnostics(
    runs: Sequence[Dict[str, Any]],
    threshold: float,
    csv_row: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    probs = np.asarray([r["probability"] for r in runs], dtype=float)
    preds = (probs >= threshold).astype(int)
    expected_score = safe_float(csv_row.get("score")) if csv_row else None
    expected_pred = csv_row.get("pred") if csv_row else None
    nearest_run = None
    if expected_score is not None and len(probs):
        nearest_idx = int(np.argmin(np.abs(probs - expected_score)))
        nearest_run = {
            "run_index": nearest_idx,
            "run_seed": runs[nearest_idx]["run_seed"],
            "probability": float(probs[nearest_idx]),
            "absolute_difference": float(abs(probs[nearest_idx] - expected_score)),
        }
    return {
        "num_resamples": len(runs),
        "probabilities": probs.tolist(),
        "mean": float(probs.mean()),
        "std": float(probs.std()),
        "min": float(probs.min()),
        "max": float(probs.max()),
        "median": float(np.median(probs)),
        "positive_prediction_rate": float(preds.mean()),
        "csv_expected": csv_row,
        "csv_score_minus_mc_mean": (
            float(expected_score - probs.mean()) if expected_score is not None else None
        ),
        "csv_score_inside_mc_range": (
            bool(probs.min() <= expected_score <= probs.max()) if expected_score is not None else None
        ),
        "agreement_rate_with_csv_pred": (
            float(np.mean(preds == int(expected_pred))) if expected_pred is not None else None
        ),
        "nearest_resample_to_csv_score": nearest_run,
        "interpretation": (
            "The data transform performs stochastic substructure subsampling. "
            "Therefore one explanation draw need not reproduce the exact score saved "
            "during the original multi-worker test evaluation. MC aggregation is "
            "used for a stability-oriented case-study visualization."
        ),
    }


# =============================================================================
# Drawing
# =============================================================================


def score_colormap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "ddi_atom_attention",
        [
            (0.10, 0.25, 0.85),
            (0.10, 0.75, 0.92),
            (0.96, 0.90, 0.18),
            (0.96, 0.42, 0.08),
            (0.70, 0.02, 0.02),
        ],
    )


def score_to_rgb(score: float) -> Tuple[float, float, float]:
    rgba = score_colormap()(float(np.clip(score, 0.0, 1.0)))
    return float(rgba[0]), float(rgba[1]), float(rgba[2])


def compute_2d(mol: Chem.Mol) -> None:
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        # MolToImage can still depict many molecules without precomputed coords.
        pass


def _configure_rdkit_paper_options(
    opts: Any,
    visual_scale: float,
    *,
    add_atom_indices: bool = True,
    substructure_mode: bool = False,
) -> None:
    """Make RDKit labels, indices, bonds and highlights readable in paper figures."""
    s = max(float(visual_scale), 0.5)
    opts.addAtomIndices = bool(add_atom_indices)
    opts.padding = max(0.035, 0.070 / max(s, 1.0))
    opts.fillHighlights = True
    opts.continuousHighlight = False
    if hasattr(opts, "atomHighlightsAreCircles"):
        opts.atomHighlightsAreCircles = True
    if hasattr(opts, "fixedFontSize"):
        opts.fixedFontSize = int(round(23 * s))
    if hasattr(opts, "minFontSize"):
        opts.minFontSize = int(round(12 * s))
    if hasattr(opts, "maxFontSize"):
        opts.maxFontSize = int(round(34 * s))
    if hasattr(opts, "annotationFontScale"):
        opts.annotationFontScale = min(1.15, 0.68 * s)
    if hasattr(opts, "bondLineWidth"):
        opts.bondLineWidth = max(2, int(round(2.2 * s)))
    if hasattr(opts, "highlightBondWidthMultiplier"):
        opts.highlightBondWidthMultiplier = max(8, int(round(8 * s)))
    if substructure_mode and hasattr(opts, "highlightBondWidthMultiplier"):
        opts.highlightBondWidthMultiplier = max(9, int(round(9 * s)))


def rdkit_attention_image(
    smiles: str,
    atom_scores: Sequence[float],
    size: Tuple[int, int] = (1100, 720),
    add_atom_indices: bool = True,
    visual_scale: float = 1.25,
):
    """Draw a per-atom blue→red attribution map with larger paper-size labels.

    Chemical bonds remain normal black/element-colored bonds; only atom halos encode
    attribution. This avoids implying a bond-level attribution that was not computed.
    """
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    compute_2d(mol)

    scores = normalize_scores(atom_scores)
    if len(scores) != mol.GetNumAtoms():
        raise ValueError(
            f"Atom score count {len(scores)} != RDKit atom count {mol.GetNumAtoms()}"
        )

    atoms = list(range(mol.GetNumAtoms()))
    colors = {i: score_to_rgb(float(scores[i])) for i in atoms}
    radii = {i: 0.27 + 0.17 * float(scores[i]) for i in atoms}

    s = max(float(visual_scale), 0.5)
    canvas = (int(round(size[0] * s)), int(round(size[1] * s)))
    drawer = rdMolDraw2D.MolDraw2DCairo(canvas[0], canvas[1])
    opts = drawer.drawOptions()
    _configure_rdkit_paper_options(
        opts, s, add_atom_indices=add_atom_indices, substructure_mode=False
    )
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms,
        highlightBonds=[],
        highlightAtomColors=colors,
        highlightAtomRadii=radii,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def draw_atom_attention(
    smiles: str,
    atom_scores: Sequence[float],
    out_png: Path,
    title: str,
    dpi: int,
    visual_scale: float = 1.25,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    s = max(float(visual_scale), 0.5)
    image = rdkit_attention_image(smiles, atom_scores, visual_scale=s)
    fig, ax = plt.subplots(figsize=(10.0 * s, 7.0 * s), dpi=dpi)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title, fontsize=16.0 * s, pad=11 * s, fontweight="semibold")
    scalar = ScalarMappable(norm=Normalize(0, 1), cmap=score_colormap())
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, fraction=0.036, pad=0.018)
    cbar.set_label(
        "Atom attribution (within-molecule normalized)",
        fontsize=11.5 * s,
        labelpad=10 * s,
    )
    cbar.ax.tick_params(labelsize=9.5 * s, width=1.0)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


def rdkit_substructure_image(
    smiles: str,
    atom_indices_to_highlight: Sequence[int],
    size: Tuple[int, int] = (720, 520),
    visual_scale: float = 1.25,
):
    """Draw one selected substructure with larger publication-style labels."""
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    compute_2d(mol)

    atoms = sorted(
        i for i in set(map(int, atom_indices_to_highlight)) if 0 <= i < mol.GetNumAtoms()
    )
    aset = set(atoms)
    bonds = [
        b.GetIdx()
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in aset and b.GetEndAtomIdx() in aset
    ]

    highlight = (0.92, 0.20, 0.16)
    atom_colors = {i: highlight for i in atoms}
    bond_colors = {i: highlight for i in bonds}
    atom_radii = {i: 0.44 for i in atoms}

    s = max(float(visual_scale), 0.5)
    canvas = (int(round(size[0] * s)), int(round(size[1] * s)))
    drawer = rdMolDraw2D.MolDraw2DCairo(canvas[0], canvas[1])
    opts = drawer.drawOptions()
    _configure_rdkit_paper_options(
        opts, s, add_atom_indices=True, substructure_mode=True
    )
    drawer.DrawMolecule(
        mol,
        highlightAtoms=atoms,
        highlightBonds=bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def draw_top_substructures_panel(
    smiles: str,
    selected: Sequence[Dict[str, Any]],
    out_png: Path,
    title: str,
    dpi: int,
    visual_scale: float = 1.25,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not selected:
        warnings.warn(f"No substructures available for {title}")
        return

    s = max(float(visual_scale), 0.5)
    cols = len(selected)
    fig, axes = plt.subplots(1, cols, figsize=(6.1 * cols * s, 5.5 * s), dpi=dpi)
    if cols == 1:
        axes = [axes]

    for rank, (ax, sub) in enumerate(zip(axes, selected), start=1):
        ax.imshow(rdkit_substructure_image(smiles, sub["atom_indices"], visual_scale=s))
        ax.axis("off")
        ax.set_title(
            f"S{rank} | {sub['type_label']} | n={sub['num_atoms']}\n"
            f"appearance={sub['appearance_rate']:.0%} | "
            f"stable-attn={sub['stability_score']:.3f}\n"
            f"atom-support={sub['mean_member_atom_attribution']:.3f}",
            fontsize=10.8 * s,
            pad=7 * s,
            fontweight="medium",
        )

    fig.suptitle(title, fontsize=16.0 * s, y=0.985, fontweight="semibold")
    fig.text(
        0.5,
        0.012,
        "Red highlight denotes substructure membership; it is not a 0-1 attention color scale.",
        ha="center",
        fontsize=9.5 * s,
    )
    fig.subplots_adjust(top=0.82, bottom=0.09, wspace=0.08)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


def draw_pair_atom_panel(
    run0: Dict[str, Any],
    atom1: Dict[str, np.ndarray],
    atom2: Dict[str, np.ndarray],
    prob_diag: Dict[str, Any],
    out_png: Path,
    dpi: int,
    visual_scale: float = 1.25,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not run0.get("smiles1") or not run0.get("smiles2"):
        return

    s = max(float(visual_scale), 0.5)
    img1 = rdkit_attention_image(run0["smiles1"], atom1["plot_score"], visual_scale=s)
    img2 = rdkit_attention_image(run0["smiles2"], atom2["plot_score"], visual_scale=s)
    fig, axes = plt.subplots(1, 2, figsize=(15.0 * s, 6.6 * s), dpi=dpi)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    for ax in axes:
        ax.axis("off")
    axes[0].set_title(
        f"Drug A: {run0['drug1']}", fontsize=14.5 * s, fontweight="medium", pad=8 * s
    )
    axes[1].set_title(
        f"Drug B: {run0['drug2']}", fontsize=14.5 * s, fontweight="medium", pad=8 * s
    )

    csv_score = None
    if prob_diag.get("csv_expected"):
        csv_score = prob_diag["csv_expected"].get("score")
    subtitle = (
        f"MC probability={prob_diag['mean']:.4f} ± {prob_diag['std']:.4f} "
        f"(n={prob_diag['num_resamples']})"
    )
    if csv_score is not None:
        subtitle += f" | saved test score={float(csv_score):.4f}"
    fig.suptitle(
        f"Atom attribution (MC-stable)\n{subtitle}",
        fontsize=15.5 * s,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(top=0.83, wspace=0.035)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


# =============================================================================
# Output tables
# =============================================================================


def write_mc_runs_csv(path: Path, runs: Sequence[Dict[str, Any]], threshold: float) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_index",
                "run_seed",
                "probability",
                "pred",
                "logit",
                "num_subtokens1",
                "num_subtokens2",
                "smoke_diff",
                "padding_bug1",
                "padding_bug2",
            ],
        )
        writer.writeheader()
        for i, r in enumerate(runs):
            writer.writerow(
                {
                    "run_index": i,
                    "run_seed": r["run_seed"],
                    "probability": r["probability"],
                    "pred": int(r["probability"] >= threshold),
                    "logit": r["logit"],
                    "num_subtokens1": r["num_subtokens1"],
                    "num_subtokens2": r["num_subtokens2"],
                    "smoke_diff": r["smoke_diff"],
                    "padding_bug1": r["padding_diag1"]["suspected_padding_mask_bug"],
                    "padding_bug2": r["padding_diag2"]["suspected_padding_mask_bug"],
                }
            )


def write_atom_aggregate_csv(
    path: Path,
    drug_id: str,
    smiles: Optional[str],
    agg: Dict[str, np.ndarray],
) -> None:
    symbols = atom_symbols(smiles)
    order = np.argsort(-agg["mean"])
    rank = {int(idx): int(pos + 1) for pos, idx in enumerate(order)}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "drug_id",
                "atom_index",
                "symbol",
                "rank",
                "mean_normalized_attribution",
                "std_normalized_attribution",
                "plot_score",
                "min_normalized_attribution",
                "max_normalized_attribution",
            ],
        )
        writer.writeheader()
        for i in range(len(agg["mean"])):
            writer.writerow(
                {
                    "drug_id": drug_id,
                    "atom_index": i,
                    "symbol": symbols[i] if i < len(symbols) else "?",
                    "rank": rank[i],
                    "mean_normalized_attribution": float(agg["mean"][i]),
                    "std_normalized_attribution": float(agg["std"][i]),
                    "plot_score": float(agg["plot_score"][i]),
                    "min_normalized_attribution": float(agg["min"][i]),
                    "max_normalized_attribution": float(agg["max"][i]),
                }
            )


def write_substructure_aggregate_csv(
    path: Path,
    drug_id: str,
    ranked: Sequence[Dict[str, Any]],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "drug_id",
                "rank",
                "type_label",
                "type_id",
                "num_atoms",
                "atom_indices",
                "appearance_count",
                "appearance_rate",
                "mean_raw_score",
                "std_raw_score",
                "mean_within_substructure_score",
                "mean_member_atom_attribution",
                "stability_score",
            ],
        )
        writer.writeheader()
        for rank, s in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "drug_id": drug_id,
                    "rank": rank,
                    "type_label": s["type_label"],
                    "type_id": s["type_id"],
                    "num_atoms": s["num_atoms"],
                    "atom_indices": " ".join(map(str, s["atom_indices"])),
                    "appearance_count": s["appearance_count"],
                    "appearance_rate": s["appearance_rate"],
                    "mean_raw_score": s["mean_raw_score"],
                    "std_raw_score": s["std_raw_score"],
                    "mean_within_substructure_score": s["mean_within_substructure_score"],
                    "mean_member_atom_attribution": s["mean_member_atom_attribution"],
                    "stability_score": s["stability_score"],
                }
            )


# =============================================================================
# Main
# =============================================================================


def explain_case(args: Any) -> None:
    cfg = convert_namespace_to_omegaconf(args)
    base_seed = int(getattr(cfg.common, "seed", 1))
    set_all_seeds(base_seed)

    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)

    checkpoint_path = args.explain_checkpoint or os.path.join(args.save_dir, "checkpoint_best.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    load_checkpoint_into_model(model, checkpoint_path, cfg.model)

    use_cpu = bool(getattr(cfg.common, "cpu", False))
    device = torch.device("cuda" if torch.cuda.is_available() and not use_cpu else "cpu")
    use_fp16 = bool(getattr(cfg.common, "fp16", False)) and device.type == "cuda"
    model.to(device)
    if use_fp16:
        model.half()
    model.eval()

    task.load_dataset(args.explain_split)
    dataset = task.dataset(args.explain_split)
    if not (0 <= args.explain_index < len(dataset)):
        raise IndexError(
            f"explain-index={args.explain_index} outside [0,{len(dataset)-1}]"
        )

    smiles_table = read_smiles_table(args.smiles_csv)
    type_map = build_type_label_map(args.id_type, args.ks)

    runs: List[Dict[str, Any]] = []
    print("\n========== SGT-DDI paper case-study explanation ==========")
    print(f"checkpoint: {checkpoint_path}")
    print(f"split/index: {args.explain_split}/{args.explain_index}")
    print(f"MC resamples: {args.mc_samples}")
    print(f"natural context batch size: {args.explain_context_batch_size}")

    for run_idx in range(args.mc_samples):
        run_seed = base_seed + run_idx * args.mc_seed_stride
        result = run_once(
            model=model,
            dataset=dataset,
            target_index=args.explain_index,
            context_batch_size=args.explain_context_batch_size,
            run_seed=run_seed,
            device=device,
            use_fp16=use_fp16,
            attention_method=args.attention_method,
            smoke_tolerance=args.smoke_test_tolerance,
            strict_smoke=args.strict_smoke_test,
            smiles_table=smiles_table,
            type_map=type_map,
            zero_epsilon=args.attention_zero_epsilon,
        )
        runs.append(result)
        print(
            f"  run {run_idx+1:02d}/{args.mc_samples}: seed={run_seed} "
            f"p={result['probability']:.6f} "
            f"subtokens=({result['num_subtokens1']},{result['num_subtokens2']})"
        )

    run0 = runs[0]
    # All resamples must refer to the same drug pair.
    for r in runs[1:]:
        if (r["drug1"], r["drug2"]) != (run0["drug1"], run0["drug2"]):
            raise RuntimeError("Drug IDs changed across resamples; dataset indexing is inconsistent.")

    ckpt_state = checkpoint_training_state(checkpoint_path)
    csv_row, csv_resolution = resolve_prediction_csv(
        args,
        ckpt_state,
        args.explain_index,
        run0["drug1"],
        run0["drug2"],
    )
    prob_diag = probability_diagnostics(runs, args.prediction_threshold, csv_row)

    # Normalized atom attribution used for the existing heatmaps.
    atom1 = aggregate_atoms(runs, "norm_atom_scores1")
    atom2 = aggregate_atoms(runs, "norm_atom_scores2")

    # PRE-min-max atom attention-rollout values.
    # These are exported only for quantitative concentration analysis;
    # the existing figures continue to use the normalized atom scores above.
    raw_atom1 = aggregate_atoms(runs, "raw_atom_scores1")
    raw_atom2 = aggregate_atoms(runs, "raw_atom_scores2")

    subs1 = aggregate_substructures(runs, "substructures1")
    subs2 = aggregate_substructures(runs, "substructures2")
    selected1 = select_nonredundant_substructures(
        subs1, args.top_k_substructures, args.max_substructure_jaccard
    )
    selected2 = select_nonredundant_substructures(
        subs2, args.top_k_substructures, args.max_substructure_jaccard
    )

    # Representative run for raw attention NPZ: closest to MC mean, NOT selected
    # to maximize confidence or match the CSV.
    probs = np.asarray([r["probability"] for r in runs], dtype=float)
    rep_idx = int(np.argmin(np.abs(probs - probs.mean())))
    representative = runs[rep_idx]

    print_padding_diagnostic(run0["drug1"], run0["padding_diag1"])
    print_padding_diagnostic(run0["drug2"], run0["padding_diag2"])
    print("\n[atom alignment]")
    print(
        f"  {run0['drug1']}: exact_order_verified={run0['alignment1'].get('exact_order_verified')} "
        f"| edge_source={run0['alignment1'].get('edge_source')} "
        f"| graph_bonds={run0['alignment1'].get('graph_atom_edge_count')} "
        f"| rdkit_bonds={run0['alignment1'].get('rdkit_bond_count')}"
    )
    print(
        f"  {run0['drug2']}: exact_order_verified={run0['alignment2'].get('exact_order_verified')} "
        f"| edge_source={run0['alignment2'].get('edge_source')} "
        f"| graph_bonds={run0['alignment2'].get('graph_atom_edge_count')} "
        f"| rdkit_bonds={run0['alignment2'].get('rdkit_bond_count')}"
    )
    print("\n[sampling stability]")
    print(
        f"  MC p = {prob_diag['mean']:.6f} ± {prob_diag['std']:.6f}; "
        f"range=[{prob_diag['min']:.6f},{prob_diag['max']:.6f}]"
    )
    if csv_row:
        print(
            f"  saved CSV p = {csv_row.get('score')} | pair_match={csv_row.get('pair_match')} | "
            f"inside_MC_range={prob_diag.get('csv_score_inside_mc_range')}"
        )
        print(f"  prediction CSV: {csv_resolution.get('selected_path')}")
    else:
        print(
            "  saved prediction CSV: not resolved; pass --prediction-csv explicitly "
            "or --prediction-csv-dir <result-directory>."
        )

    target_value = run0["target"]
    aggregate_prediction = prediction_summary(target_value, prob_diag["mean"], args.prediction_threshold)

    out_dir = Path(args.explain_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"case_{args.explain_split}_{args.explain_index}_{run0['drug1']}_{run0['drug2']}"

    result_json: Dict[str, Any] = {
        "checkpoint": checkpoint_path,
        "checkpoint_training_state": ckpt_state,
        "prediction_csv_resolution": csv_resolution,
        "split": args.explain_split,
        "index": args.explain_index,
        "base_seed": base_seed,
        "device": str(device),
        "fp16_inference": use_fp16,
        "attention_method": args.attention_method,
        "drug1": run0["drug1"],
        "drug2": run0["drug2"],
        "smiles1": run0["smiles1"],
        "smiles2": run0["smiles2"],
        "target": target_value,
        "aggregate_prediction": aggregate_prediction,
        "sampling_probability_diagnostic": prob_diag,
        "representative_run_index": rep_idx,
        "representative_run_seed": representative["run_seed"],
        "representative_run_probability": representative["probability"],
        "normal_vs_explain_logit_max_abs_diff_across_runs": max(r["smoke_diff"] for r in runs),
        "atom_alignment1": run0["alignment1"],
        "atom_alignment2": run0["alignment2"],
        "padding_diagnostic1": run0["padding_diag1"],
        "padding_diagnostic2": run0["padding_diag2"],
        # Existing within-molecule normalized atom attribution.
        "drug1_atom_mc_mean": atom1["mean"].tolist(),
        "drug1_atom_mc_std": atom1["std"].tolist(),
        "drug1_atom_plot_score": atom1["plot_score"].tolist(),
        "drug2_atom_mc_mean": atom2["mean"].tolist(),
        "drug2_atom_mc_std": atom2["std"].tolist(),
        "drug2_atom_plot_score": atom2["plot_score"].tolist(),

        # PRE-min-max raw attention-rollout atom scores.
        # These are the quantities needed to determine whether the apparent
        # localization exists before visualization normalization.
        "drug1_atom_raw_mc_mean": raw_atom1["mean"].tolist(),
        "drug1_atom_raw_mc_std": raw_atom1["std"].tolist(),
        "drug1_atom_raw_mc_min": raw_atom1["min"].tolist(),
        "drug1_atom_raw_mc_max": raw_atom1["max"].tolist(),
        "drug2_atom_raw_mc_mean": raw_atom2["mean"].tolist(),
        "drug2_atom_raw_mc_std": raw_atom2["std"].tolist(),
        "drug2_atom_raw_mc_min": raw_atom2["min"].tolist(),
        "drug2_atom_raw_mc_max": raw_atom2["max"].tolist(),

        "drug1_aggregated_substructures": subs1,
        "drug2_aggregated_substructures": subs2,
        "drug1_selected_top_substructures": selected1,
        "drug2_selected_top_substructures": selected2,
        "type_id_mapping": {str(k): v for k, v in type_map.items()},
        "visualization_notes": {
            "atom_normalization": (
                "Each stochastic run is min-max normalized over real atoms only; "
                "the figure shows the MC mean re-normalized within the molecule."
            ),
            "raw_atom_rollout": (
                "drug*_atom_raw_mc_* fields are pre-min-max CLS-to-atom attention-rollout "
                "values aggregated across MC resamples. They are attribution signals, "
                "not calibrated probabilities and are not used to recolor the existing figures."
            ),
            "substructure_ranking": (
                "stability_score = appearance_rate × mean within-substructure-token normalized attention"
            ),
            "substructure_highlight": (
                "Red denotes membership in the displayed substructure, not an attention magnitude."
            ),
            "attention_causality_warning": (
                "Attention attribution is descriptive and non-directional; it is not a causal effect."
            ),
        },
    }

    json_path = out_dir / f"{prefix}.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(tensor_to_python(result_json), handle, ensure_ascii=False, indent=2)

    write_mc_runs_csv(out_dir / f"{prefix}_mc_runs.csv", runs, args.prediction_threshold)
    write_atom_aggregate_csv(
        out_dir / f"{prefix}_{run0['drug1']}_atoms_mc.csv",
        run0["drug1"],
        run0["smiles1"],
        atom1,
    )
    write_atom_aggregate_csv(
        out_dir / f"{prefix}_{run0['drug2']}_atoms_mc.csv",
        run0["drug2"],
        run0["smiles2"],
        atom2,
    )
    write_substructure_aggregate_csv(
        out_dir / f"{prefix}_{run0['drug1']}_substructures_mc.csv", run0["drug1"], subs1
    )
    write_substructure_aggregate_csv(
        out_dir / f"{prefix}_{run0['drug2']}_substructures_mc.csv", run0["drug2"], subs2
    )

    if args.save_attention_npz:
        arrays: Dict[str, np.ndarray] = {}
        # Save only the representative MC run to avoid huge files.
        local = representative["local_batch_index"]
        for i, attn in enumerate(representative["attention1"]):
            if attn is not None:
                a = attn.detach().float().cpu()
                if a.dim() == 4:
                    a = a[:, local : local + 1]
                arrays[f"drug1_layer_{i:02d}"] = a.numpy()
        for i, attn in enumerate(representative["attention2"]):
            if attn is not None:
                a = attn.detach().float().cpu()
                if a.dim() == 4:
                    a = a[:, local : local + 1]
                arrays[f"drug2_layer_{i:02d}"] = a.numpy()
        np.savez_compressed(out_dir / f"{prefix}_representative_attention.npz", **arrays)

    if not args.skip_drawing:
        if run0["smiles1"]:
            draw_atom_attention(
                run0["smiles1"],
                atom1["plot_score"],
                out_dir / f"{prefix}_{run0['drug1']}_atom_attention_mc.png",
                f"{run0['drug1']} atom attribution (MC-stable)",
                args.paper_dpi,
                args.paper_visual_scale,
            )
            draw_top_substructures_panel(
                run0["smiles1"],
                selected1,
                out_dir / f"{prefix}_{run0['drug1']}_top_substructures_mc.png",
                f"{run0['drug1']}: stable top substructures",
                args.paper_dpi,
                args.paper_visual_scale,
            )
        if run0["smiles2"]:
            draw_atom_attention(
                run0["smiles2"],
                atom2["plot_score"],
                out_dir / f"{prefix}_{run0['drug2']}_atom_attention_mc.png",
                f"{run0['drug2']} atom attribution (MC-stable)",
                args.paper_dpi,
                args.paper_visual_scale,
            )
            draw_top_substructures_panel(
                run0["smiles2"],
                selected2,
                out_dir / f"{prefix}_{run0['drug2']}_top_substructures_mc.png",
                f"{run0['drug2']}: stable top substructures",
                args.paper_dpi,
                args.paper_visual_scale,
            )
        draw_pair_atom_panel(
            run0,
            atom1,
            atom2,
            prob_diag,
            out_dir / f"{prefix}_pair_atom_attention_mc.png",
            args.paper_dpi,
            args.paper_visual_scale,
        )

    print("\n========== Done ==========")
    print(f"JSON: {json_path}")
    print(f"Output dir: {out_dir}")
    print(f"Representative run: {rep_idx} (seed={representative['run_seed']})")
    print(
        "Recommended paper use: prefer cases with small MC probability std, high "
        "prediction agreement, exact atom alignment, and stable local substructures."
    )


def main() -> None:
    parser = options.get_training_parser()
    parser.add_argument("--explain-checkpoint", default=None)
    parser.add_argument("--explain-split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--explain-index", type=int, default=0)
    parser.add_argument("--explain-output-dir", default="case_study_outputs_paper")
    parser.add_argument("--attention-method", default="rollout", choices=["rollout", "last_cls"])
    parser.add_argument("--smiles-csv", default="get_data/idsmile.csv")
    parser.add_argument(
        "--prediction-csv",
        default=None,
        help=(
            "Saved test prediction CSV (e.g. epoch_65_test_....csv). If omitted, "
            "the script tries to auto-discover the CSV for the completed best epoch."
        ),
    )
    parser.add_argument(
        "--prediction-csv-dir",
        default=None,
        help=(
            "Directory to search recursively for epoch_*_test_*.csv when "
            "--prediction-csv is omitted."
        ),
    )
    parser.add_argument("--top-k-substructures", type=int, default=3)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)
    parser.add_argument("--attention-zero-epsilon", type=float, default=1e-10)
    parser.add_argument("--smoke-test-tolerance", type=float, default=1e-5)
    parser.add_argument("--strict-smoke-test", action="store_true")
    parser.add_argument("--save-attention-npz", action="store_true")
    parser.add_argument("--skip-drawing", action="store_true")

    # New paper/stability options.
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=8,
        help="Number of stochastic substructure resamples. Use 16-32 for final paper cases.",
    )
    parser.add_argument(
        "--mc-seed-stride",
        type=int,
        default=10007,
        help="Deterministic seed spacing between resamples.",
    )
    parser.add_argument(
        "--explain-context-batch-size",
        type=int,
        default=8,
        help=(
            "Use the target sample's natural batch context. Set to the batch size used "
            "during test evaluation (8 in the current experiment)."
        ),
    )
    parser.add_argument(
        "--max-substructure-jaccard",
        type=float,
        default=0.85,
        help="Maximum overlap allowed between displayed top substructures; 1.0 disables filtering.",
    )
    parser.add_argument("--paper-dpi", type=int, default=300)
    parser.add_argument(
        "--paper-visual-scale",
        type=float,
        default=1.25,
        help=(
            "Global scale for RDKit atom symbols/indices, bond width, matplotlib "
            "titles/labels and figure size. Default 1.25 is tuned for journal figures; "
            "try 1.35-1.45 if labels are still small."
        ),
    )

    args = options.parse_args_and_arch(parser, modify_parser=None)
    if args.mc_samples < 1:
        raise ValueError("--mc-samples must be >= 1")
    if not (0.0 <= args.max_substructure_jaccard <= 1.0):
        raise ValueError("--max-substructure-jaccard must be in [0,1]")
    if args.paper_visual_scale <= 0:
        raise ValueError("--paper-visual-scale must be > 0")
    explain_case(args)


if __name__ == "__main__":
    main()
