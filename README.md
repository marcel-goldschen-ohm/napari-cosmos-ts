# napari-cosmos-ts
[napari](https://napari.org/stable/) plugin for colocalization single-molecule spectroscopy (CoSMoS) time series (TS)

# Install
1. Install [Miniconda](https://docs.conda.io/en/main/miniconda.html). Simplest is to download and run the installer. This will get you the conda package manager.
2. Install [napari](https://napari.org/stable/). The following recommended method may change, so you should check the napari website. The following creates a sandboxed environment for napari with python, then activates that environement in conda, then installs napari into the environement (pip is automatically installed along with python). In a command shell or terminal run the following three commands:
```
conda create -y -n napari-env -c conda-forge python=3.9
conda activate napari-env
pip install "napari[all]"
```
3. Install [tifffile](https://github.com/cgohlke/tifffile), [pyqtgraph](https://www.pyqtgraph.org), [pystackreg](https://pystackreg.readthedocs.io/en/latest/readme.html#installation), and [pycpd](https://github.com/siavashk/pycpd). Note that pystackreg and pycpd are optional and only used for image and point registration. In a command shell or terminal run the following three commands:
```
conda activate napari-env
conda install -c conda-forge tifffile pyqtgraph pystackreg
pip install pycpd
```
4. Download `napari_cosmos_ts.py` from this repository.
5. Run `napari_cosmos_ts.py`. In a command shell or terminal run the following two commands:
```
conda activate napari-env
python your/path/to/napari_cosmos_ts.py
```
