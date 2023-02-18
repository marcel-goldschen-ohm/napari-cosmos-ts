# napari-cosmos-ts
[napari](https://napari.org/stable/) plugin for colocalization single-molecule spectroscopy (CoSMoS) time series (TS)

# Install
1. Install [Miniconda](https://docs.conda.io/en/main/miniconda.html). Simplest is to download and run the installer. This will get you the conda package manager.
2. Install [napari](https://napari.org/stable/). The following recommended method may change, so you should check the napari website. The following creates a sandboxed environment for napari, then activates that environement in conda, then installs napari into the environement.
```
conda create -y -n napari-env -c conda-forge python=3.9
conda activate napari-env
pip install "napari[all]"
```
3. Download `napari_cosmos_ts.py` from this repository.
