import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from rdkit import Chem
import torch.nn.functional as F
from scoring_functions import get_scoring_function
from tqdm import tqdm
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
from collections import Counter
import functools
from torch.utils.data import Dataset
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from torch.optim import AdamW
import torch
from torch.utils.data import  DataLoader

class SmilesDataset(Dataset):
    def __init__(self, file_path):

        with open(file_path, 'r') as f:
            self.smiles_list = f.read().splitlines()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        return self.smiles_list[idx]

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def filter_valid_smiles(batch_tensor):
    valid_list = []
    valid_indices = []
    oh = OneHotFeaturizer()
    sample = VAE_model.channel_decoder(batch_tensor)
    recon_x = VAE_model.decode(sample)
    recon_x = recon_x.cpu().detach().numpy()
    y = np.argmax(recon_x, axis=2)
    for i in range(y.shape[0]):
        updata_smiles = oh.decode_smiles_from_index(y[i])
        mol = Chem.MolFromSmiles(updata_smiles)
        if mol is not None:
            valid_list.append(updata_smiles)
            valid_indices.append(i)

    smiles_counter = Counter(valid_list)
    most_common_smiles = smiles_counter.most_common(10)
    top_smiles_list = [smile for smile, count in most_common_smiles]
    top_smiles_indices = [valid_indices[valid_list.index(smile)] for smile in top_smiles_list]
    top_encodings = batch_tensor[top_smiles_indices]
    top_encodings = [top_encodings[i].unsqueeze(0) for i in range(top_encodings.shape[0])]
    return top_encodings


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


def creat_dataset(batch_size = 1000,smi_file_path = "data/3000.smi"):
    oh = OneHotFeaturizer()
    smiles_dataset = SmilesDataset(smi_file_path)
    out_smiles = []
    for i in tqdm(range(len(smiles_dataset.smiles_list)), desc="Dataset preparation"):
        smiles = smiles_dataset.smiles_list[i]
        smiles = smiles.ljust(120)
        smiles = torch.from_numpy(oh.featurize([smiles]).astype(np.float32)).to('cuda')
        start_vec = smiles.transpose(1, 2).to("cuda")
        mu, logvar = VAE_model.encode(start_vec)
        std = torch.exp(0.5 * logvar)

        for j in range(200):
            eps = 4e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            w = VAE_model.channel_encoder(w)
            if j == 0:
                w_out = w
            else:
                w_out = torch.cat((w_out, w), dim=0)
        out  = filter_valid_smiles(w_out )
        out_smiles = out_smiles + out

    custom_dataset = CustomDataset(out_smiles)
    smiles_dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    return smiles_dataloader


def trainer(device):
    #VAE_model = MolecularVAE()
    VAE_model.load_state_dict(torch.load('vae_ckpt/vae-103-6.896378517150879_successful_channel.pth'))
    VAE_model.to('cuda')
    for param in VAE_model.parameters():  #
        param.requires_grad = False  #

    model, diffusion = create_model_and_diffusion()
    model.to("cuda")
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    mp_trainer = MixedPrecisionTrainer(model=model,use_fp16=False,fp16_scale_growth=0.001)
    opt = AdamW(mp_trainer.master_params, lr=0.0001, weight_decay=0.0)

    smiles_dataloader = creat_dataset()
    for step in range(0, 20001):
        loss_total = 0
        for smiles in tqdm(smiles_dataloader, desc="Training progress"):
            smiles = smiles.squeeze(dim=1)
            z = smiles
            t, weights = schedule_sampler.sample(z.shape[0], "cuda")

            compute_losses = functools.partial(diffusion.training_losses,model,z,t, model_kwargs={})
            losses = compute_losses()
            loss = (losses["loss"] * weights).mean()
            loss.requires_grad_(True)

            loss_total = loss_total + loss.item()

            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        if step % 500 == 0:
            torch.save(model.state_dict(), f"ddm_ckpt/diffusion_model_{step}.pth")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    VAE_model = MolecularVAE()
    trainer(device)





