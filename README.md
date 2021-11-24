# About Project

Video Link: https://drive.google.com/file/d/1n1jUvktJyWmPGh19Gj48zb9bFQCaKODQ/view?usp=sharing

### How to run:

- All the experiments are done in jupyter notebook. 
- Start by opening the notebook and executing the code blocks in order.
- data has to be downloaded externally and placed in the directory of choice.
- edit the files `src/common/config.py` and `src/data/config.py` to change directory structure of data.

### File descriptions

- Baseline 1: `multimodal_vae_cnn_with_poe.ipynb` is the multimodal VAE with Product of Experts (PoE).
- Baseline 2: `multimodal_vae_cnn.ipynb` is the our multimodal VAE without the Wasserstein Distance.
- Method 1 on MNIST: `multimodal_vae_cnn_with_wloss.ipynb` is the Multimodal Wasserstein VAE (MW-VAE) on MNIST.
- Method 1 on MS-COCO: `multimodal_vae_cnn_with_wloss_COCO.ipynb` is the Multimodal Wasserstein VAE (MW-VAE) on MS-COCO.
- Method 2 on MNIST: `disentangled_multimodal_vae_cnn_with_wloss.ipynb` is the implementation of Factorized MW-VAE on MNIST.

### Implementation.

- Starting Code:
    - Original Unimodal VAE (can be seen in the notebook `unimodal_vae.ipynb`) is the adaptation from original paper. 
    - Basic implementation starts with the implementation of HW assignment on VAE.
- Modified Code:
    - Encoder and Decoder architectures are changed to include the Conv and TransposeConv block instead of linear layers.
    - Multimodal VAE is extended on top of Unimodal VAE
    - Multimodal Wasserstein VAE (MW-VAE) is extended by adding Wasserstein Loss calculation is to the overall loss term.
    - Factorized MW-VAE is extended on top of MW-VAE to output factorized latent variables.
- Original Code:
    - `data/loader.py` implements the loading function and associated augmentations for all the datasets used. 
    - `data/coco.py` is implemented to do a cached loading of MS-COCO dataset. This helps in speeding up the training process.
    - `data/embedding.py` follows the Huggingface tutorial to extract text embedding using Pre-trained BERT models.

### Datasets:

- MNIST:
    - Images are considered as modality 1
    - corresponding labels are used as modality 2
    - Obtained from the official website using the torchvision.datasets: https://pytorch.org/vision/stable/datasets.html
- MS-COCO:
    - Images are considered as modality 1
    - Bert embeddings generated for the first caption of the image is used as modality 2.
    - downloaded from https://cocodataset.org/#download