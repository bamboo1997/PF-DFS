git clone https://github.com/automl/NASLib.git
rm -rf NASLib/.git*
cd NASLib
pip install --upgrade pip setuptools wheel
pip install numpy==1.22.0
pip install -e .
cd ..

pip install hpbandster
pip install dehb==0.1.1
pip install omegaconf
pip install scipy==1.4.1
pip install ConfigSpace==0.6.1
