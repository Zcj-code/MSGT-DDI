import pandas as pd
import torch
import pickle
import random
import os
from rdkit import Chem
from rdkit.Chem import Crippen, rdmolops
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

import sascorer  # sascorer.py


def atom_features(atom):

    symbol_list = [
        "C", "N", "O", "S", "F", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I",
        "B", "V", "K", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "Li", "Ge", "Au", "Cu", "Ni", "Mn", "Other"
    ]
    symbol = atom.GetSymbol()
    symbol_onehot = [int(symbol == s) for s in symbol_list]


    degree = atom.GetDegree()
    degree_onehot = [int(degree == i) for i in range(7)]


    formal_charge = atom.GetFormalCharge()
    fc_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    fc_onehot = [0] * 6
    fc_onehot[fc_map.get(formal_charge, 5)] = 1


    chiral_tag = int(atom.GetChiralTag())
    chiral_onehot = [int(chiral_tag == i) for i in range(5)]


    num_H = atom.GetTotalNumHs()
    numH_onehot = [int(num_H == i) for i in range(6)]


    hybrid = int(atom.GetHybridization())
    hybrid_onehot = [int(hybrid == i) for i in range(6)]


    aromatic = [int(atom.GetIsAromatic())]


    mass = [atom.GetMass() / 200.0]

    features = (
            symbol_onehot + degree_onehot + fc_onehot + chiral_onehot +
            numH_onehot + hybrid_onehot + aromatic + mass
    )
    return features  # 32+7+6+5+6+6+1+1=64



def bond_features(bond):

    bt = bond.GetBondType()
    bond_type_onehot = [
        int(bt == Chem.BondType.SINGLE),
        int(bt == Chem.BondType.DOUBLE),
        int(bt == Chem.BondType.TRIPLE),
        int(bt == Chem.BondType.AROMATIC)
    ]
    exists = [1]

    is_conjugated = [int(bond.GetIsConjugated())]


    is_in_ring = [int(bond.IsInRing())]

    stereo = bond.GetStereo()
    stereo_onehot = [int(stereo == i) for i in range(7)]

    features = exists + bond_type_onehot + is_conjugated + is_in_ring + stereo_onehot  # 共1+4+1+1+7=14维
    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if mol.GetNumAtoms() <= 7:
        return None

    num_atom = mol.GetNumAtoms()
    atom_feats = []
    for i in range(num_atom):
        atom_feats.append(atom_features(mol.GetAtomWithIdx(i)))
    atom_feats = torch.tensor(atom_feats, dtype=torch.float)  # [num_atoms, 64]


    adj = torch.zeros((num_atom, num_atom), dtype=torch.int8)
    bond_feats = torch.zeros((num_atom, num_atom, 14), dtype=torch.float)

    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = 1
        bf = bond_features(bond)
        bond_feats[i, j] = torch.tensor(bf)
        bond_feats[j, i] = torch.tensor(bf)


        edge_indices.append([i, j])
        edge_indices.append([j, i])
        edge_attrs.append(bf)
        edge_attrs.append(bf)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()  # [2, num_edges]
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)  # [num_edges, 14]

    return {"atom_feats": atom_feats, "adj": adj, "bond_feats": bond_feats}


def process_dataset(pairs_file, id_to_smiles, dataset_name, output_dir="processed_0802"):

    os.makedirs(output_dir, exist_ok=True)


    df_pairs = pd.read_csv(pairs_file)

    samples = []
    failed = 0
    skipped_large = 0
    skipped_db00080 = 0

    print(f"\n处理数据集: {dataset_name}")
    print(f"原始样本数: {len(df_pairs)}")

    for idx, row in tqdm(df_pairs.iterrows(), total=len(df_pairs)):
        drug_a, drug_b, label = row["drug_a"], row["drug_b"], row["label"]


        if drug_a not in id_to_smiles or drug_b not in id_to_smiles:
            failed += 1
            continue

        smi1 = id_to_smiles[drug_a]
        smi2 = id_to_smiles[drug_b]





        mol1 = smiles_to_graph(smi1)
        mol2 = smiles_to_graph(smi2)

        if mol1 is None or mol2 is None:
            failed += 1
            continue


        if mol1['atom_feats'].shape[0] > 128 or mol2['atom_feats'].shape[0] > 128:
            skipped_large += 1
            continue

        sample = {
            "x1": mol1["atom_feats"],  # [num_atoms1, 64]
            "x2": mol2["atom_feats"],  # [num_atoms2, 64]
            "adj1": mol1["adj"],  # [num_atoms1, num_atoms1]
            "adj2": mol2["adj"],  # [num_atoms2, num_atoms2]
            "bond_feats1": mol1["bond_feats"],  # [num_atoms1, num_atoms1, 14]
            "bond_feats2": mol2["bond_feats"],  # [num_atoms2, num_atoms2, 14]
            "y": int(label),
            "global_idx1": drug_a,
            "global_idx2": drug_b,
        }
        samples.append(sample)

    print(f"✅ 处理完成: {dataset_name}")
    print(f"  有效样本数: {len(samples)}")
    print(f"  跳过无效样本: {failed}")
    print(f"  跳过大分子: {skipped_large}")
    print(f"  跳过DB00080: {skipped_db00080}")


    def adjust_to_multiple_of_8(data):
        remainder = len(data) % 8
        if remainder != 0:
            return data[:-remainder]
        return data

    samples = adjust_to_multiple_of_8(samples)
    print(f"  调整后样本数: {len(samples)}")


    def save_pickle(data, name):
        with open(f"{output_dir}/{name}.pickle", "wb") as f:
            pickle.dump(data, f)
        with open(f"{output_dir}/{name}.index", "w") as f:
            f.write(",".join(map(str, range(len(data)))))
        print(f"  📦 保存: {name}.pickle 和 {name}.index")

    save_pickle(samples, dataset_name)


    def save_samples_to_csv(data, name):
        csv_data = []
        for sample in data:

            csv_data.append({
                "global_idx1": sample["global_idx1"],
                "global_idx2": sample["global_idx2"],
                "label": sample["y"],
                "x1_feats": ";".join([",".join([str(f) for f in atom.tolist()]) for atom in sample["x1"]]),
                "x2_feats": ";".join([",".join([str(f) for f in atom.tolist()]) for atom in sample["x2"]]),
            })

        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(f"{output_dir}/{name}_full_samples.csv", index=False)
        print(f"  📄 保存完整样本数据: {name}_full_samples.csv")

    save_samples_to_csv(samples, dataset_name)

    return samples


def main():
    # 读取id到smiles的映射
    print("读取id到smiles的映射...")
    df_idsmile = pd.read_csv("idsmile.csv")

    # 创建id到smiles的字典
    id_to_smiles = {}
    for _, row in df_idsmile.iterrows():
        id_to_smiles[row["id"]] = row["smiles"]

    print(f"成功加载 {len(id_to_smiles)} 个药物的SMILES")

    # 处理三个数据集
    datasets = [
        ("train_pairs.csv", "train"),
        ("val_pairs.csv", "val"),
        ("test_pairs.csv", "test")
    ]


    for file_name, _ in datasets:
        if not os.path.exists(file_name):
            print(f"错误: 找不到文件 {file_name}")
            return

    all_samples = {}

    for pairs_file, dataset_name in datasets:
        samples = process_dataset(pairs_file, id_to_smiles, dataset_name, "processed_0802")
        all_samples[dataset_name] = samples


    print("\n" + "=" * 50)
    print("数据预处理完成！")
    print("=" * 50)
    for name in ["train", "val", "test"]:
        print(f"{name}: {len(all_samples[name])} 个样本")

    print(f"\n输出目录: processed_0802/")
    print("生成的文件:")
    for name in ["train", "val", "test"]:
        print(f"  - {name}.pickle")
        print(f"  - {name}.index")
        print(f"  - {name}_full_samples.csv")


if __name__ == "__main__":

    main()