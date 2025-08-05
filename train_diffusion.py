import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from rdkit import Chem
import torch.nn.functional as F
from tqdm import tqdm
from scoring_functions import get_scoring_function
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
import torch
import functools
from torch.utils.data import Dataset, DataLoader
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from torch.optim import AdamW


class SmilesDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            self.smiles_list = f.read().splitlines()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx]


def filter_valid_smiles(batch_tensor, device,VAE_model):
    valid_indices = []
    index = 0
    oh = OneHotFeaturizer()
    # VAE_model = MolecularVAE().to(device)
    sample = VAE_model.channel_decoder(batch_tensor)
    recon_x = VAE_model.decode(sample)
    recon_x = recon_x.cpu().detach().numpy()
    y = np.argmax(recon_x, axis=2)
    for i in range(y.shape[0]):
        updata_smiles = oh.decode_smiles_from_index(y[i])
        mol = Chem.MolFromSmiles(updata_smiles)
        if mol is not None:
            valid_indices.append(index)
        index = index + 1

    filtered_tensor = batch_tensor[valid_indices]
    return filtered_tensor, valid_indices


def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def creat_dataset(batch_size=1000, smi_file_path="data/250k_rndm_zinc_drugs_clean.smi"):
    oh = OneHotFeaturizer()
    smiles_dataset = SmilesDataset(smi_file_path)

    for i in range(len(smiles_dataset.smiles_list)):
        smiles = smiles_dataset.smiles_list[i]
        smiles = smiles.ljust(120)
        smiles = torch.from_numpy(oh.featurize([smiles]).astype(np.float32)).to('cuda')
        smiles = smiles.squeeze(0)
        smiles_dataset.smiles_list[i] = smiles

    smiles_dataloader = DataLoader(smiles_dataset, batch_size=batch_size, shuffle=True)
    return smiles_dataloader


def trainer(device):
    VAE_model = MolecularVAE()
    VAE_model.load_state_dict(torch.load('vae_ckpt/vae-103-6.896378517150879_successful_channel.pth'))
    VAE_model.to('cuda')
    for param in VAE_model.parameters():  #
        param.requires_grad = False  #

    model, diffusion = create_model_and_diffusion()
    model.to("cuda")
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)
    mp_trainer = MixedPrecisionTrainer(model=model, use_fp16=False, fp16_scale_growth=0.001)
    opt = AdamW(mp_trainer.master_params, lr=0.0001, weight_decay=0.0)


    smiles_dataloader = creat_dataset()

    for step in range(0, 10001):
        loss_total = 0
        total = 0
        for smiles in tqdm(smiles_dataloader, desc="Training progress"):
            total = total +1
            start_vec = smiles.transpose(1, 2).to(device)
            mu, logvar = VAE_model.encode(start_vec)
            std = torch.exp(0.5 * logvar)

            eps = 3e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)

            w = VAE_model.channel_encoder(w)
            w, index = filter_valid_smiles(w, device,VAE_model)
            z = w
            #print(z.shape[0])
            t, weights = schedule_sampler.sample(z.shape[0], "cuda")

            compute_losses = functools.partial(diffusion.training_losses, model, z, t, model_kwargs={})
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()
            loss.requires_grad_(True)
            loss_total = loss_total + loss.item()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
        print("step: ", step + 1)
        print("loss: ", loss_total / total)

        if step % 200 == 0:
            torch.save(model.state_dict(), f"ddm_ckpt/diffusion_model_{step}.pth")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer(device)



