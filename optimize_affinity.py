import torch as th
import numpy as np
import torch.nn.functional as F
from guided_diffusion.resample import create_named_schedule_sampler
from torch.optim import AdamW
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from rdkit import Chem
from typing import List
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion
)
import sys
from torch import nn, optim
from tqdm import tqdm
from joblib import Parallel
from expert import ga_operator
import pandas as pd

from affinity.affinity_score import pred_affinity

from polygon.utils.utils import canonicalize_list
from scoring_functions import get_scoring_function
import torch
from affinity.priority_queue import MaxRewardPriorityQueue
import functools
from torch.utils.data import Dataset, DataLoader
from guided_diffusion.fp16_util import MixedPrecisionTrainer


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


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

def GenerateSamples():
    list_smiles = []
    results: List[OptResult] = []
    total_smiles = []
    total_scores = []
# freeze
    for param in model.parameters():  #
        param.requires_grad = False  #
    # sample
    for param in MolecularVAE.parameters():  #
        param.requires_grad = False  #

    sample = sample_fn(
        model,
        (10, 3, 16, 16),
        clip_denoised=True,
        model_kwargs={},
        device="cuda",
        progress=True
    )
    chunk_tensors = torch.chunk(sample, 10)

    for sample in chunk_tensors:
        sample = MolecularVAE.channel_decoder(sample)
        recon_x = MolecularVAE.decode(sample)
        recon_x = recon_x.cpu().detach().numpy()
        y = np.argmax(recon_x, axis=2)
        updata_smiles = oh.decode_smiles_from_index(y[0])
        list_smiles.append(updata_smiles)

# Check for reasonableness
    canonicalized_smiles = list(canonicalize_list(list_smiles, include_stereocenters=True))

    # -----------------------------------
    scores_structure_start = scoring_function(canonicalized_smiles)
    scores_structure_start = scores_structure_start.tolist()

    temp_results = [OptResult(smiles=smiles, score=score) for smiles, score in
                    zip(canonicalized_smiles, scores_structure_start)]
    temp_results = sorted(temp_results, reverse=True)
    temp = 0
    temp_unique_results = {}
    for result in temp_results:
        smiles = result.smiles
        score = result.score
        if smiles not in temp_unique_results:  # or score > unique_results[smiles]:
            temp = temp + 1
            temp_unique_results[smiles] = score
        if temp == 30:
            break
    canonicalized_smiles = list(temp_unique_results.keys())
    # ---------------------------------------------

# Calculate affinity score
    scores = pred_affinity(canonicalized_smiles, "hybrid/data/raw/gen_smi.csv")
    scores = scores.tolist()
    start_smiles = canonicalized_smiles
    start_scores = scores

    apprentice_storage.add_list(smis=start_smiles, scores=start_scores)
    apprentice_storage.squeeze_by_kth(k=4)

    for step in tqdm(range(4)):
        expert_smis, expert_scores = apprentice_storage.sample_batch(2)

        gen_smiles = expert_handler.query(
            query_size=100, mating_pool=expert_smis, pool=pool
        )

        gen_smiles = list(canonicalize_list(gen_smiles, include_stereocenters=True))
        # -----------------------------------
        scores_structure = scoring_function(gen_smiles)
        scores_structure = scores_structure.tolist()

        temp_results_1 = [OptResult(smiles=smiles, score=score) for smiles, score in
                          zip(gen_smiles, scores_structure)]
        temp_results_1 = sorted(temp_results_1, reverse=True)
        temp = 0
        temp_unique_results = {}
        for result in temp_results_1:
            smiles = result.smiles
            score = result.score
            if smiles not in temp_unique_results:  # or score > unique_results[smiles]:
                temp = temp + 1
                temp_unique_results[smiles] = score
            if temp == 50:
                break
        gen_smiles = list(temp_unique_results.keys())
        scores_structure = list(temp_unique_results.values())

        gen_scores = pred_affinity(gen_smiles, "hybrid/data/raw/gen_smi.csv")
        gen_scores = gen_scores.tolist()

        total_smiles = total_smiles + gen_smiles
        total_scores = total_scores + gen_scores

    int_results = [OptResult(smiles=smiles, score=score) for smiles, score in
                   zip(canonicalized_smiles + total_smiles, scores + total_scores)]
    int_results = sorted(int_results, reverse=True)
    temp = 0
    unique_results = {}
    for result in int_results:
        smiles = result.smiles
        score = result.score
        if smiles not in unique_results:  # or score > unique_results[smiles]:
            temp = temp + 1
            unique_results[smiles] = score
        if temp == 40:
            break
    subset = list(unique_results.keys())
    score = list(unique_results.values())

    df = pd.DataFrame({'SMILES': subset, 'score': score})
    df.to_csv('out_put.csv', index=False)

    return subset

def OptimizeVAE(train_dataloader, total_epoch):
    print("------------------------------------------")
    print("Optimize the weight of VAE generation model")
    for param in MolecularVAE.parameters():  #
        param.requires_grad = True  #

    torch.manual_seed(42)
    epochs = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer_VAE = optim.Adam(MolecularVAE.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    flag = 0
    for epoch in range(1, epochs + 1):
        batch_size_vae = 40
        train_loss = 0
        train_channel_loss = 0
        for batch_idx, smiles in enumerate(train_dataloader):
            start = smiles[0]
            start = start.ljust(120)
            start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to('cuda')

            for i in range(1, batch_size_vae):
                start_1 = smiles[i]
                start_1 = start_1.ljust(120)
                start_vec_1 = torch.from_numpy(oh.featurize([start_1]).astype(np.float32)).to('cuda')
                start_vec = torch.cat([start_vec, start_vec_1], dim=0)
            start_vec = start_vec.transpose(1, 2).to("cuda")

            optimizer_VAE.zero_grad()
            recon_batch, mu, logvar, channel_x = MolecularVAE(start_vec)
            mu_1 = mu.detach()
            channel_loss = criterion(mu_1, channel_x)
            loss_1 = loss_function(recon_batch, start_vec.transpose(1, 2), mu, logvar)
            loss = loss_1 + channel_loss

            loss.backward()
            train_loss += loss_1
            train_channel_loss += channel_loss
            optimizer_VAE.step()
        if train_loss / len(train_dataloader.dataset) < 1.5:
            print(epoch)
            flag = 1
            break

    if flag == 0:
        sys.exit("VAE model optimization error, program exits")

    torch.save(MolecularVAE.state_dict(), 'op_vae/vae_similty_{}.pth'.format(total_epoch))

def OptimizeDDM(train_dataloader, total_epoch):
    print("------------------------------------------")
    print("optimizing the weights of the diffusion model")
    flag = 0
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    mp_trainer = MixedPrecisionTrainer(
        model=model,
        use_fp16=False,
        fp16_scale_growth=0.001,
    )
    opt = AdamW(
        mp_trainer.master_params, lr=0.0001, weight_decay=0.0  # 0.00005
    )

    batch_size_diffusion = 40

    for param in MolecularVAE.parameters():  #
        param.requires_grad = False  #
    for param in model.parameters():  #
        param.requires_grad = True  #

    for step in range(0, 10001):
        if step %100 == 0:
            print(step)
        loss_total = 0
        for smiles in (train_dataloader):
            start = smiles[0]
            start = start.ljust(120)
            start_vec = torch.from_numpy(oh.featurize([start]).astype(np.float32)).to('cuda')

            for i in range(1, batch_size_diffusion):
                start_1 = smiles[i]
                start_1 = start_1.ljust(120)
                start_vec_1 = torch.from_numpy(oh.featurize([start_1]).astype(np.float32)).to('cuda')
                start_vec = torch.cat([start_vec, start_vec_1], dim=0)

            start_vec = start_vec.transpose(1, 2).to("cuda")
            mu, logvar = MolecularVAE.encode(start_vec)
            z = MolecularVAE.channel_encoder(mu)

            t, weights = schedule_sampler.sample(batch_size_diffusion, "cuda")
            compute_losses = functools.partial(diffusion.training_losses,model,z,t,model_kwargs={})
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()
            loss.requires_grad_(True)

            loss_total = loss_total + loss.item()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        if loss_total / 1 < 0.0018:
            print(step)
            flag = 1
            break
    if flag == 0:
        sys.exit("Diffusion model optimization error, program exits")

    torch.save(model.state_dict(), 'op_diffusion/diffusion_similty_{}.pth'.format(total_epoch))



if __name__ == '__main__':
    import sys

    # 只保留 sys.argv[0]，清除 Jupyter 传递的 -f 参数
    if '-f' in sys.argv:
        sys.argv = sys.argv[:1]
    scoring_function_kwargs = {}
    scoring_function = get_scoring_function(scoring_function="tanimoto", num_processes=0, **scoring_function_kwargs)

    oh = OneHotFeaturizer()
    MolecularVAE = MolecularVAE()
    MolecularVAE.load_state_dict(torch.load('op_vae/vae-103-6.896378517150879_successful_channel.pth'))
    MolecularVAE.to('cuda')
    for param in MolecularVAE.parameters():  #
        param.requires_grad = False  #

    model, diffusion = create_model_and_diffusion()
    model.load_state_dict(th.load("op_diffusion/model_successful_1000.pth"))  # 加载训练好的模型
    model.to("cuda")
    # model.eval()
    for param in model.parameters():  #
        param.requires_grad = False  #

    sample_fn = diffusion.p_sample_loop

    apprentice_storage = MaxRewardPriorityQueue()
    expert_storage = MaxRewardPriorityQueue()
    expert_handler = ga_operator(mutation_rate=0.05)  # 突变概率
    pool = Parallel(n_jobs=8)

    for epoch in range(10):
        print("epoch： ",epoch)
        print("Sampling optimization training dataset")
        train_set = GenerateSamples()
        train_dataloader = DataLoader(train_set, batch_size=40, shuffle=True)

        OptimizeVAE(train_dataloader, epoch)
        OptimizeDDM(train_dataloader, epoch)











