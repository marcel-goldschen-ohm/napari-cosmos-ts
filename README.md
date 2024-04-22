# napari-cosmos-ts
[napari](https://napari.org/stable/) plugin for colocalization single-molecule spectroscopy (CoSMoS) time series (TS) analysis.

# Install
1. Install the `conda` package manager. Simplest is to download [Miniconda](https://docs.conda.io/en/main/miniconda.html) and run the installer.
2. Create a virtual python environment named `napari-env` (or name it whatever you want) in which to install [napari](https://napari.org/stable/) and this plugin. In a command shell or terminal run the following command:
```shell
conda create --name napari-env python
```
3. Activate your virtual environment. *!!! Note you will have to do this every time you open a new command shell or terminal.* In a command shell or terminal run the following command:
```shell
conda activate napari-env
```
4. Install `napari` and `napari-cosmos-ts` into your virtual environment. In a command shell or terminal *where you have activated your virtual environment* run the following command:
```shell
pip install "napari[all]" napari-cosmos-ts
```
Or for the latest version of `napari-cosmos-ts`:
```shell
pip install "napari[all]" napari-cosmos-ts@git+https://github.com/marcel-goldschen-ohm/napari-cosmos-ts
```

# Run
1. Activate your virtual environment (see [Install](#install), replace napari-env with the name of your environment). In a command shell or terminal run the following command:
```shell
conda activate napari-env
```
2. Launch the `napari` viewer. In a command shell or terminal *where you have activated your virtual environment* run the following command:
```shell
napari
```
3. Launch the `napari-cosmos-ts` plugin. From the napari viewer `Plugins menu`, select `Colocalization Single-Molecule Time Series (napari-cosmos-ts)`. This should bring up a docked widget within the viewer. **Now you are good to go!**

# User Guide
:construction: