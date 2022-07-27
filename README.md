# PyTorch semantic segmentation example code for Liver segmentation (Medical Decalthon Dataset)

## Environment installation

Download and install mambaforge (faster miniforge/mniniconda alternative) from below link
```
https://github.com/conda-forge/miniforge#mambaforge
```

```bash
mamba create -n liver-segmentation python=3.9 -c conda-forge
conda activate liver-segmentation
mamba install numpy matplotlib jupyterlab tqdm qudida scikit-image scipy pyyaml scikie-learn pywavelets tifffile imageio networkx threadpoolctl joblib dicom2nifti  -c conda-forge -y
mamba install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge -y
pip install opencv-python-headless albumentations
pip install git+https://github.com/qubvel/segmentation_models.pytorch.git
```

## Dataset Preparation

Download Task03_Liver.tar to dataset/ from http://medicaldecathlon.com and extract by using the following command

```
tar -xvf dataset/Task03_Liver.tar
```
Then run dataPrepation.py to convert niff file to png format

PS. If you used your lab server, the prepared dataset is located at "/mnt/datasets/liver"

## Training
See detail and adjust parameters to suited your need in train.py
```
python train.py
```
Training output (weight/output_mask) will be placed in
```
outputs/{expName}/
```