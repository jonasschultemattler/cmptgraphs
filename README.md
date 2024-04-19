# Introduction

**This is an open-source repository contributing Methods to compare Temporal Graphs and their empirical evaluation.**

It contains all code used in experiments in [thesis](https://github.com/jonasschultemattler/cmptgraphs/blob/master/main.pdf)
and includes an efficient adapted implementation of Dynamic Temporal Graph Warping [Froese et. al](http://dx.doi.org/10.1007/s13278-020-00664-5) in C with a Python binding.


## Compilation

### Compile dtgw
```
cd code/dtgw

make

cd ../..
```

### Setup Python

Create virtual environment
```
mkdir venv

python3 -m venv venv

source venv/bin/activate
```


Install python modules
```
python3 -m pip install -r req.txt

export PYTHONPATH=.
```


## Usage



### Download datasets

```
chmod +x getds.sh

./getds.sh
```


### Run Experiments

```
mkdir output

mkdir output/brains

mkdir output/dissemintaion

cd code

./proximity_dissemination.sh

./train_dissemination.sh

./proximity_brains.sh

./train_brains.sh
```


## Contact

Feel free to use the code and contact me for questions.


Please cite our [thesis](https://github.com/jonasschultemattler/cmptgraphs/blob/master/main.pdf), and the papers of the methods used, if you use our implementations:

```
@Thesis{schultemattler2023cmptgraphs,
  title={Methods for Comparing Temporal Graphs: An empirical Study},
  author={Schulte-Mattler Jonas},
  type={mathesis},
  year={2023},
  institution={TU Berlin}
}
```

## License
Released under MIT license.
See [LICENSE.md](LICENSE.md) for details.



