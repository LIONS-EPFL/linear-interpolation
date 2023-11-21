# Stable Nonconvex-Nonconcave Training via Linear Interpolation

This official code for [Stable Nonconvex-Nonconcave Training via Linear Interpolation](https://arxiv.org/pdf/2310.13459.pdf) accepted at NeurIPS 2023.

## Setup

```
conda create -n rapp python=3.8
conda activate rapp

# On GPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt
python setup.py develop
source .env
python rapp/runner.py -h
```

For wandb:

- Delete wandb entry from `/home/<user>/.netrc` if present to prevent auto-login to a different account
- Storage your key with `vim .env`:
  ```bash
  export WANDB_API_KEY=<mykey>
  ```
- Before running a script run `source .env`


## Usage

1. This code contains 13 optimizers on CIFAR-10 listed in Table 2&3 in the paper. Use `--opt RAPP` to modify it.

    | --opt     | GDA    | EG     | EGplus | LA-GDA | LA-EG  | LA-EGplus | RAPP   |
    |-----------|--------|--------|--------|--------|--------|-----------|--------|
    | optimizer | GDA    | EG     | EG+    | LA-GDA | LA-EG  | LA-EG+    | RAPP   |

    | --opt     | Adam    | EA     | EAplus | LA-Adam | LA-EA  | LA-EAplus | 
    |-----------|--------|--------|--------|--------|--------|-----------|
    | optimizer | Adam  | ExtraAdam | ExtraAdam+ | LA-Adam | LA-ExtraAdam | LA-ExtraAdam+ | 

2. Example scripts:

    ```python
    python rapp/runner.py --model resnet --dataset cifar10 --loss hinge --spectral-norm --batch-size 128 --epochs 3200 --lrG 0.02 --lrD 0.1 --nz 128 --ngf 128 --ndf 64 --opt RAPP --num-D-step 1 --num-metrics-samples 50000 --wandb-name "cifar10/RAPP/hinge/resnet/bs128/net64/lr(0.02_0.1)/eval50000/anchor_change(multi3_lam0.9)" --gpus 1 --workers 200 --inner-steps 3 --lam 0.9
    ```


## Citation
```
@inproceedings{pethick2023stable,
  title={Stable Nonconvex-Nonconcave Training via Linear Interpolation},
  author={Pethick, Thomas and Xie, Wanyun and Cevher, Volkan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```