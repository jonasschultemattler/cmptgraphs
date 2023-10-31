## Compile dtgw

cd code/dtgw

make

cd ../..


## Create virtual environment

mkdir venv

python3 -m venv venv

source venv/bin/activate


## Install python modules

python3 -m pip install -r req.txt

export PYTHONPATH=.


## Download datasets

chmod +x getds.sh

./getds.sh



## Run Experiments

mkdir output
mkdir output/brains
mkdir output/dissemintaion

cd code

./proximity_dissemination.sh
./train_dissemination.sh
./proximity_brains.sh
./train_brains.sh

