# AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization

<img src="image/pytorch-logo-dark.png" width="10%"/> [![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="image/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


Paper accepted at the [NAACL-HLT 2021](https://2021.naacl.org):

**[AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization](https://arxiv.org/pdf/2103.11332)**, by **[Tiezheng Yu*](https://tysonyu.github.io/)**, **[Zihan Liu*](https://zliucr.github.io/)**, [Pascale Fung](https://pascale.home.ece.ust.hk).

## Abstract
State-of-the-art abstractive summarization models generally rely on extensive labeled data, which lowers their generalization ability on domains where such data are not available. In this paper, we present a study of domain adaptation for the abstractive summarization task across six diverse target domains in a low-resource setting. Specifically, we investigate the second phase of pre-training on large-scale generative models under three different settings: 1) source domain pre-training; 2) domain-adaptive pre-training; and 3) task-adaptive pre-training. Experiments show that the effectiveness of pre-training is correlated with the similarity between the pre-training data and the target domain task. Moreover, we find that continuing pre-training could lead to the pre-trained model's catastrophic forgetting, and a learning method with less forgetting can alleviate this issue. Furthermore, results illustrate that a huge gap still exists between the low-resource and high-resource settings, which highlights the need for more advanced domain adaptation methods for the abstractive summarization task.

## Dataset
We release the AdaptSum dataset, which contains the summarization datasets across six target domains as well as the corpora for SDPT, DAPT and TAPT. You can download AdaptSum from [Here](https://drive.google.com/drive/folders/1qdkavIQonTAepkJhGpo3TZpU4LUW44sp?usp=sharing).

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
- We take `cnn_dm` as an example
1. Create a new folder named `SDPT_save` at the root of this project
2. Prepare dataloader:
    ```
    python ./src/preprocessing.py -data_path=dataset/ \
                            -data_name=SDPT-cnn_dm \
                            -mode=train \
                            -batch_size=4
    ```
3. Run `./scripts/sdpt_pretraining.sh`. You can add `-recadam` and `-logging_Euclid_dist` to use RecAdam.

## DAPT pretraining
- We take `debate domain` as an example
1. Create a new folder named `DAPT_save` at the root of this project
2. Run `./scripts/dapt_pretraining.sh`. You can add `-recadam` and `-logging_Euclid_dist` to use RecAdam.

## TAPT pretraining
- We take `debate domain` as an example
1. Create a new folder named `TAPT_save` at the root of this project
2. Run `./scripts/tapt_pretraining.sh`. You can add `-recadam` and `-logging_Euclid_dist` to use RecAdam.

## Fine-tuning
- We take `debate domain` as an example
1. Create a new folder named `debate` at `logs`

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

3. Install `pyrouge` package (You can skip this if you have already installed `pyrouge`)
    - Step 1 : Install Pyrouge from source (not from pip)
    ```
    git clone https://github.com/bheinzerling/pyrouge
    cd pyrouge
    pip install -e .
    ```
    - Step 2 : Install official ROUGE script
    ```
    git clone https://github.com/andersjo/pyrouge.git rouge
    ```
    - Step 3 : Point Pyrouge to official rouge script (The path given to pyrouge should be absolute path !)
    ```
    pyrouge_set_rouge_path ~/pyrouge/rouge/tools/ROUGE-1.5.5/
    ```
    - Step 4 : Install libxml parser
    As mentioned in this [issue](https://github.com/bheinzerling/pyrouge/issues/27), you need to install libxml parser
    ```
    sudo apt-get install libxml-parser-perl
    ```
    - Step 5 : Regenerate the Exceptions DB
    As mentioned in this [issue](https://github.com/bheinzerling/pyrouge/issues/8), you need to regenerate the Exceptions DB
    ```
    cd rouge/tools/ROUGE-1.5.5/data
    rm WordNet-2.0.exc.db
    ./WordNet-2.0-Exceptions/buildExeptionDB.pl ./WordNet-2.0-Exceptions ./smart_common_words.txt ./WordNet-2.0.exc.db
    ```
    - Step 6 : Run the tests
    ```
    python -m pyrouge.test
    ```

4. Run Finetuning
    - If you don't want to use any second phase of pre-training, run:
        ```
        python ./src/run.py -visible_gpu=0 \
                            -data_name=debate  \
                            -save_interval=100 \
                            -start_to_save_iter=3000
        ```
    - If you want to use pretrained checkpoints from **SDPT**, run:
        ```
        python ./src/run.py -visible_gpu=0 \
                            -data_name=debate \
                            -save_interval=100 \
                            -start_to_save_iter=3000 \
                            -pre_trained_src \
                            -train_from=YOUR_SAVED_CHECKPOINTS
        ```
    - If you want to use pretrained checkpoints from **DAPT** or **TAPT**, run:
        ```
        python ./src/run.py -visible_gpu=0 \
                            -data_name=debate \
                            -save_interval=100 \
                            -start_to_save_iter=3000 \
                            -pre_trained_lm=YOUR_SAVED_CHECKPOINTS
        ```

5. Evaluate the performance
    1) Make a folder named `inference` at `logs`
    2) You can do inference by
        ```
        python ./src/inference.py -visible_gpu=0 -train_from=YOUR_SAVED_CHECKPOINT
        ```
    3) You can calculate rouge scores by
        ```
        python ./src/cal_roug.py -c=CANDIDATE_FILE -r=REFERENCE_FILE -p=NUMBER_OF_PROCESS
        ```

## References
If you use our benchmark or the code in this repo, please cite our paper.

<pre>
@inproceedings{Yu2021AdaptSum,
  title={AdaptSum: Towards Low-Resource Domain Adaptation for Abstractive Summarization},
  author={Tiezheng Yu and Zihan Liu and Pascale Fung},
  journal={arXiv preprint arXiv:2103.11332},
  year={2021}
}
</pre>

Also, please consider citing all the individual datasets in your paper.

Dialog domain:
<pre>
@inproceedings{gliwa2019samsum,
  title={SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization},
  author={Gliwa, Bogdan and Mochol, Iwona and Biesek, Maciej and Wawer, Aleksander},
  booktitle={Proceedings of the 2nd Workshop on New Frontiers in Summarization},
  pages={70--79},
  year={2019}
}
</pre>

Email domain:
<pre>
@inproceedings{zhang2019email,
  title={This Email Could Save Your Life: Introducing the Task of Email Subject Line Generation},
  author={Zhang, Rui and Tetreault, Joel},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={446--456},
  year={2019}
}
</pre>

Movie and debate domains:
<pre>
@inproceedings{wang2016neural,
  title={Neural Network-Based Abstract Generation for Opinions and Arguments},
  author={Wang, Lu and Ling, Wang},
  booktitle={Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
  pages={47--57},
  year={2016}
}
</pre>

Social media domain:
<pre>
@inproceedings{kim2019abstractive,
  title={Abstractive Summarization of Reddit Posts with Multi-level Memory Networks},
  author={Kim, Byeongchang and Kim, Hyunwoo and Kim, Gunhee},
  booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)},
  pages={2519--2531},
  year={2019}
}
</pre>

Science domain:
<pre>
@inproceedings{yasunaga2019scisummnet,
  title={Scisummnet: A large annotated corpus and content-impact models for scientific paper summarization with citation networks},
  author={Yasunaga, Michihiro and Kasai, Jungo and Zhang, Rui and Fabbri, Alexander R and Li, Irene and Friedman, Dan and Radev, Dragomir R},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  pages={7386--7393},
  year={2019}
}
</pre>

