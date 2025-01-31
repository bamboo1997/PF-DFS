conda install -y pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y numpy=1.23.5 -c conda-forge
conda install -y pandas==2.2.1 -c conda-forge
conda install -y matplotlib=3.8.3 -c conda-forge
conda install -y albumentations=1.3.1 -c conda-forge
conda install -y optuna=3.5.0 -c conda-forge
conda install -y cmaes=0.10.0 -c conda-forge
conda install -y plotly=5.19.0 -c conda-forge
conda install -y nvitop=1.3.2 -c conda-forge
conda install -y seaborn=0.13.2 -c conda-forge
conda install -y omegaconf=2.3.0 -c conda-forge
conda install -y black=24.2.0 -c conda-forge
conda install -y isort=5.13.2 -c conda-forge
conda install -y click=8.1.3 -c conda-forge
conda install -y moviepy=1.0.3 -c conda-forge

pip install kaleido==0.2.1
pip install hpbandster
pip install dehb==0.1.1
pip install segmentation-models-pytorch

pip install "git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym"
git clone https://github.com/slds-lmu/yahpo_data.git
rm -rf yahpo_data/.git*
mv yahpo_data/ hpo_benchmarks
