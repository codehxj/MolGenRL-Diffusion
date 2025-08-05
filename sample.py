
import torch as th
import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from affinity.affinity_score import pred_affinity
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
import re
from polygon.utils.utils import canonicalize_list
import csv
import torch

class OptResult:
    def __init__(self, smiles: str, score: float) -> None:
        self.smiles = smiles
        self.score = score
    def __eq__(self, other):
        return (self.score, self.smiles) == (other.score, other.smiles)
    def __lt__(self, other):
        return (self.score, self.smiles) < (other.score, other.smiles)

def validate_smiles(smiles):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return 0
        return 1
    except:
        return 0

def sample_smiles():
    sample_fn = diffusion.p_sample_loop
    smiles_list = []

    sample = sample_fn(
        model,
        (20, 3, 16, 16),
        clip_denoised=True,
        model_kwargs={},
        device="cuda",
        progress=True
    )
    chunk_tensors = torch.chunk(sample, 20)
    for sample in chunk_tensors:
        sample = prior_model.channel_decoder(sample)
        recon_x = prior_model.decode(sample)
        recon_x = recon_x.cpu().detach().numpy()
        y = np.argmax(recon_x, axis=2)
        updata_smiles = oh.decode_smiles_from_index(y[0])
        smiles_list.append(updata_smiles)

    smiles_list = list(canonicalize_list(smiles_list, include_stereocenters=True))
    print(len(smiles_list))
    for i in range(len(smiles_list)):
        print("{}".format(smiles_list[i]))

if __name__ == "__main__":
    oh = OneHotFeaturizer()
    prior_model = MolecularVAE()
    prior_model.load_state_dict(torch.load("vae_ckpt/vae-103-6.896378517150879_successful_channel.pth"))
    prior_model.to('cuda')
    prior_model.eval()

    model, diffusion = create_model_and_diffusion()
    model.load_state_dict(th.load("ddm_ckpt/model_successful_1000.pth"))
    model.to("cuda")
    model.eval()

    smiles_gen = sample_smiles()
    # Calculate_affinity(smiles_gen)
    # Similarity(smiles_gen)
    # stability(smiles_gen)












