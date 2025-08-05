## Paper Data

The datasets used in this project can be found at the following links:

- **Training dataset for training molecular generative models:**

  - The [**ChEMBL**](https://chembl.gitbook.io/chembl-interface-documentation/downloads) dataset used in this project can be found here.  
  - The [**QM9**](https://drive.google.com/file/d/1JZ_Z5bjS0RsX_BRWtrplMN9vZpL78-T7/view?usp=drive_link) dataset used in this project can be found here.  
  - The [**GEOM-Drug**](https://dataverse.harvard.edu/file.xhtml?fileId=4360331&version=2.0) dataset used in this project can be found here.  
  - The [**ZINC**](https://drive.google.com/file/d/1N44fpvCKEqI3xorXH7Q9sOq2f4ylCUwz/view) dataset used in this project can be found here.  

## Training

1. Train the encoder

   ```bash
   python train_vae.py

2. Training diffusion model
   
    ```bash
   python train_diffusion.py

## Sample

  Sampling molecules using generative models.

    python sample.py

## Optimize

Please run the optimize_affinity.py and optimize_similarity.py files to optimize the generative model for specific tasks.



