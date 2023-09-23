# LEATHER-AACL2022
This is the repository for ["LEATHER: A Framework for Learning to Generate Human-like Text in Dialogue"](https://arxiv.org/pdf/2210.07777v1.pdf) to appear in [AACL 2022](https://www.aacl2022.org).

## Package
A package to compute the discrete energy statistic (proposed in the paper) can be found [here](https://github.com/anthonysicilia/discrete-energy).

## Dependencies
This repository is heavily based the [code repo](https://github.com/shekharRavi/Beyond-Task-Success-NAACL2019) for the NAACL19 paper ["Beyond task success: A closer look at jointly learning to see, ask, and GuessWhat"](https://arxiv.org/abs/1809.03408). 

Our updates consist of (1) added human regularization loss (discussed in the paper), (2) changes to the evaluation protocol (also discussed in the paper), (3) code to conduct human evaluation and compute our proposed energy statistic (discussed in the paper), and (4) some adversarial losses not discussed in the paper. Note, we also updated code throughout to be compatible with newer version of PyTorch. 

Data access and additional tips for running can be found at the [code repo](https://github.com/shekharRavi/Beyond-Task-Success-NAACL2019) for "Beyond Task Success" as well.

## Running the Code
Examples for training and running inference are provided in bash scripts within the `scripts` directory; e.g., `prep.sh` runs the supervised pre-training phase, `train-*.sh` runs different versions of the CL training phase, and `inference-*.sh` runs different versions of the inference phase. Logs from the inference phase are analyzed using the `main.py` script from the `analysis` directory. The `scripts` directory also contains other various scripts used to conduct the human evaluation and finalize results for the paper (e.g., compute energy and produce energy plots).

If you use our code and/or ideas from our paper, please consider citing us (and the authors of our dependencies).

My contact information is available in the paper if you run into any issues with the code. Alternatively, feel free to raise an issue on this repo.

### Notable Version
The code was run using the following versions:
 - h5py==3.6.0
 - matplotlib==3.5.0
 - nltk==3.6.5
 - numpy==1.21.2
 - pandas==1.3.5
 - Pillow==9.3.0
 - scikit_learn==1.1.3
 - tensorboardX==2.5.1
 - torch==1.10.2 (build info py3.7_cuda10.2_cudnn7.6.5_0)
 - torchvision==0.11.3
 
 ## More Papers
 This paper is one of a series from our lab using learning theory to study understanding and generation in NLP. Check out some of our other papers here:
  - [The Change that Matters in Discourse Parsing: Estimating the Impact of Domain Shift on Parser Error](https://arxiv.org/abs/2203.11317)
  - [PAC-Bayesian Domain Adaptation Bounds for Multiclass Learners](https://openreview.net/forum?id=S0lx6I8j9xq)
  - [Modeling Non-Cooperative Dialogue: Theoretical and Empirical Insights](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00507/113020/Modeling-Non-Cooperative-Dialogue-Theoretical-and)
  - [Learning to Generate Equitable Text in Dialogue from Biased Training Data](https://github.com/anthonysicilia/equitable-dialogue-ACL2023/blob/main/README.md)

