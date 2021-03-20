# AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization

<img src="image/pytorch-logo-dark.png" width="10%"/> [![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="image/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


Paper accepted at the [NAACL-HLT 2021](https://2021.naacl.org):

**AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization**, by **[Tiezheng Yu](https://tysonyu.github.io/)**, **[Zihan Liu](https://zliucr.github.io/)**, [Pascale Fung](https://pascale.home.ece.ust.hk).

If your work is inspired by our paper, or you use any code snippets in this repo, please cite this paper, the BibTex is shown below:

<pre>

</pre>

## Abstract
State-of-the-art abstractive summarization models generally rely on extensive labeled data, which lowers their generalization ability on domains where such data are not available. In this paper, we present a study of domain adaptation for the abstractive summarization task across six diverse target domains in a low-resource setting. Specifically, we investigate the second phase of pre-training on large-scale generative models under three different settings: 1) source domain pre-training; 2) domain-adaptive pre-training; and 3) task-adaptive pre-training. Experiments show that the effectiveness of pre-training is correlated with the similarity between the pre-training data and the target domain task. Moreover, we find that continuing pre-training could lead to the pre-trained model's catastrophic forgetting, and a learning method with less forgetting can alleviate this issue. Furthermore, results illustrate that a huge gap still exists between the low-resource and high-resource settings, which highlights the need for more advanced domain adaptation methods for the abstractive summarization task.

## Dataset
We release all the data for SDPT, DAPT and TAPT on google drive. [Here](https://drive.google.com/drive/folders/1qdkavIQonTAepkJhGpo3TZpU4LUW44sp?usp=sharing) is the link for AdaptSum.

## Preparation for running
1. Create a new folder named `dataset` at the root of this project
2. Download the data from google drive and then put it in the `dataset` folder
3. Create the conda environment
```
conda create -n adaptsum python=3.6
```
4. Activate the conda environment
```
conda activate adaptsum
```
5. Install pytorch. Please check your CUDA version before the installation and modify it accordingly, or you can refer to [pytorch website](https://pytorch.org)
```
conda install pytorch cudatoolkit=11.0 -c pytorch
```
6. Install requirements
```
pip install -r requirements.txt
```
7. Create a new folder named `logs` at the root of this project
## SDPT pretraining
### here we take `cnn_dm` as an example
1. Create a new folder named `SDPT_save` at the root of this project
2. Prepare dataloader:
```
python ./src/preprocessing.py -data_path=dataset/ \
                        -data_name=SDPT-cnn_dm \
                        -mode=train \
                        -batch_size=4
```
3. Run `./scripts/sdpt_pretraining.sh`. If you want to use recadam uncomment `-recadam` and `-logging_Euclid_dist`

## DAPT pretraining
### here we take `debate domain` as an example
1. Create a new folder named `DAPT_save` at the root of this project
2. Run `./scripts/dapt_pretraining.sh`. If you want to use recadam uncomment `-recadam` and `-logging_Euclid_dist`

## TAPT pretraining
### here we take `debate domain` as an example
1. Create a new folder named `TAPT_save` at the root of this project
2. Run `./scripts/tapt_pretraining.sh`. If you want to use recadam uncomment `-recadam` and `-logging_Euclid_dist`

## Finetune
### here we take `debate domain` as an example
1. Create a new folder named `Finetune_save` at the root of this project
2. Prepare dataloader:
```
python ./src/preprocessing.py -data_path=dataset/ \
                        -data_name=debate \
                        -mode=train \
                        -batch_size=4
python ./src/preprocessing.py -data_path=dataset/ \
                        -data_name=debate \
                        -mode=valid \
                        -batch_size=4
python ./src/preprocessing.py -data_path=dataset/ \
                        -data_name=debate \
                        -mode=test \
                        -batch_size=4
```
3.
- If you don't want to use any pretrained model, run:
```
python ./src/run.py -visible_gpu=4 -data_name=debate  -start_to_save_iter=500
```
- If you want to use