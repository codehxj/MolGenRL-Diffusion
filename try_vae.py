import numpy as np
from featurizer import OneHotFeaturizer
from models import MolecularVAE
from polygon.utils.utils import canonicalize_list
import torch

if __name__ == '__main__':

    oh = OneHotFeaturizer()
    MolecularVAE = MolecularVAE()
    MolecularVAE.load_state_dict(torch.load('vae_ckpt/vae-103-6.896378517150879_successful_channel.pth'))
    MolecularVAE.to('cuda')
    for param in MolecularVAE.parameters():  #
        param.requires_grad = False  #
    subset = ["COC(=O)c1ccc(NC(=O)NCC(C)(C)c2ccncc2)cc1C"]

    smile_list = []
    for j in range(10):
        for i in range(len(subset)):
            #print(i)
            smiles_test = subset[i]
            start_vec = torch.from_numpy(oh.featurize([smiles_test]).astype(np.float32)).to('cuda')
            start_vec = start_vec.transpose(1, 2).to("cuda")
            mu, logvar = MolecularVAE.encode(start_vec)
            w = mu
            z = MolecularVAE.channel_encoder(w)
            w = MolecularVAE.channel_decoder(z)

            recon_x = MolecularVAE.decode(w)
            recon_x = recon_x.cpu().detach().numpy()
            y = np.argmax(recon_x, axis=2)

            end = oh.decode_smiles_from_index(y[0])

            smile_list.append(end)

    canonicalized_smiles = list(canonicalize_list(smile_list, include_stereocenters=True))
    print(len(canonicalized_smiles))
    for i in range(len(canonicalized_smiles)):
        print(canonicalized_smiles[i])
