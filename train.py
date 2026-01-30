import sys
from rdkit import Chem
import graph_tool
import torch_geometric
import joblib

import warnings

warnings.filterwarnings('ignore')
import fairseq


from fairseq_cli.train import cli_main
import logging
from torch_geometric.datasets import ZINC
logging.getLogger().setLevel(logging.INFO)



if __name__ == "__main__":
    cli_main()