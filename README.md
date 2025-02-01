# README

This repository is the code base for Parameter-Free Dynamic Fidelity Selection.

## Create environment.

We employ Miniconda as our Python runtime environment.
The execution environment related to NASBench201 is independent of the package management.
The following describes how to create environments other than NASBench201 and only for NASBench201.

### LCBench, NASBench301, and DNNs

The python version of this environment is `3.10`.

1. create and activate the virtual environment.

```
   conda create -n pf-dfs python=3.10
   conda activate pf-dfs
```

2. execute `script/setup.sh` to install the required python package.

### NASBench201

The python version od this enviroment is `3.8`.

1. create and activate the virtual environment.

```
   conda create -n pf-dfs-naslib python=3.8
   conda activate pf-dfs-naslib
```

2. execute `script/setup_naslib.sh` to install the required python package.


## Prepare dataset

The training datasets used in the DNN, CIFAR10 and CamVid, are ceated as follows.

#### CIFAR100

1. Execute `python etc/create_cifar100.py`.

2. Execute `tar -I pigz -cf cifar100.tar.gz cifar100/`.

#### Camvid

1. Download CamVid dataset as follow link.  
   "https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid"

2. Execute `mkdir datasets` and move the download CamVid dataset to `./datasets`.

3. Rename the download CamVid dataset to "CamVid_org".

4. Execute `python etc/convert_CamVid_dataset.py`.

5. Execute `tar -I pigz -cf CamVid.tar.gz CamVid/`.


## Usage

In this repository, hyperparameter optimization is available for LCBench and DNNs using Pytorch, and neural architecture search is available for NASBench201 and NASBench301.  
The execution command and available optimization algorithms in each problem are as follows.
The basic execution command is `python src/search.py <config_name>`.

## Example


---
If running the configuration file as is,
```
python src/search.py lcbench
```

If running a configuration file with overwriting settings,
```
python src/search.py lcbench sampler.name=CMAES sampler.n_trials=100
```

## Citations
For the citation, use the following format:

```
@article{takenaga2025,
   title = {Parameter-free dynamic fidelity selection for automated machine learning},
   journal = {Applied Soft Computing},
   pages = {112750},
   year = {2025},
   issn = {1568-4946},
   doi = {https://doi.org/10.1016/j.asoc.2025.112750},
   url = {https://www.sciencedirect.com/science/article/pii/S1568494625000614},
   author = {Shintaro Takenaga and Yoshihiko Ozaki and Masaki Onishi},
}
```