# SACHA

The code implementation for SACHA: Soft Actor-Critic with Heuristic-Based Attention for Partially Observable Multi-Agent Path Finding.

![Model Design](https://raw.githubusercontent.com/Qiushi-Lin/SACHA/master/figures/model_design.png)

## Setup

**Dependencies**

Create the virtual environment and install the required packages.
```
conda create -n sacha python=3.10
pip install -r requirement.txt
conda activate sacha
```

**Benchmarks**

Generate the test set used for evaluation.
```
cd benchmarks
python create_test.py
```

## Train

**SACHA**

  ``python train.py``

**SACHA(C)**

  ``python train.py --communication``

## Evaluate

  ``python evaluate.py --load_from_dir path/to/dir``

# Contact

Email: qiushi_lin@sfu.ca
