## Introduction
This is the Pytorch implementation for our paper at KDD'23: **Knowledge Graph Self-Supervised Rationalization for Recommendation**.

## Environment Dependencies
You can refer to `requirements.txt` for the experimental environment we set to use.

## run KGRec
Simply use:

`python run_kgrec.py --dataset [dataset_name]`

And the hyperparameters we use are fixed according to the dataset in `KGRec.py`.

## Baseline Models (KGCL, KGIN)
We also implement KGCL and include the original KGIN release in our repository. For example, to run KGCL, you may execute:

**alibaba-ifashion**

`python run_kgcl.py --mu 0.7 --tau 0.2 --cl_weight 0.1`

**last-fm**

`python run_kgcl.py --mu 0.5 --tau 0.1 --cl_weight 0.1`

**mind**

`python run_kgcl.py --mu 0.6 --tau 0.2 --cl_weight 0.1`

## Citation
Please kindly cite our work if you find our paper or codes helpful.
```
@inproceedings{yang2023knowledge,
  title={Knowledge graph self-supervised rationalization for recommendation},
  author={Yang, Yuhao and Huang, Chao and Xia, Lianghao and Huang, Chunzhen},
  booktitle={Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining},
  pages={3046--3056},
  year={2023}
}
```
