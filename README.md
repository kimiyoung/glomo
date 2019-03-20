# GLoMo

## Introduction

This repo contains the code we used in our paper:

> [Unsupervised Learning of Transferable Relational Graphs](https://papers.nips.cc/paper/8110-glomo-unsupervised-learning-of-transferable-relational-graphs)
>
> Zhilin Yang*, Jake Zhao*, Bhuwan Dhingra, Kaiming He, William W. Cohen, Ruslan Salakhutdinov, Yann LeCun
> NeurIPS 2018 (*: equal contribution)

## Requirements

The project was developed in early 2018, so we used pytorch 0.3.1.
Adaptations to the latest pytorch version are welcome through PRs.

Python 3, pytorch 0.3.1, and spacy.

## Unsupervised Pretraining

Copy the data needed for pretraining from our server.
```
cd pretrain
./copy_data.sh
```

Train the model

`
python main.py --dec_len 3 --nlayers 3 --nlayers_key 4 --nlayers_query 4 --cnn_sa 1 --no_gru 0 --yann 1 --use_value 1 --sparse_mode 4 --topk_rate 1
`

## Transfer Learning for SQuAD 1.1

Download the data necessary for finetuning:
```
cd squad
./download.sh
./copy_data.sh
```

### Transfer Learning with GloVe

`
python main.py --mode train --no_bucket --keep_prob 0.8 --ptr_keep_prob 0.8 --batch_size 32 --hidden0 76 --hidden1 76 --keep_prob0 0.8 --use_glove --pre_att_id SKIP-v16-cnn-gru-yann-ly3-k4q4v1-lr8.0-sp4-topk1.0-declen3 --original_ptr --condition 1 --gate_fuse 3 --att_cnt 2
`

### Transfer Learning with TRNN

TRNN means applying the learned graph structures on the hidden layers instead of the input embeddings. (See paper for more details.)

`
python main.py --mode train --no_bucket --keep_prob 0.8 --ptr_keep_prob 0.8 --batch_size 32 --hidden0 76 --hidden1 76 --keep_prob0 0.8 --use_glove --pre_att_id SKIP-v16-cnn-gru-yann-ly3-k4q4v1-lr8.0-sp4-topk1.0-declen3 --original_ptr --condition 1 --gate_fuse 3 --att_cnt 2 --trnn
`

### Working with ELMo

In our paper, we also had experiments of applying GLoMo on top of ELMo.

You will first need to extract elmo features by doing `python main.py --mode elmo` along with other arguments that fit your needs.

Then you can finetune with or without TRNN as follows

`
python main.py --mode train --no_bucket --keep_prob 0.8 --ptr_keep_prob 0.8 --batch_size 32 --hidden0 76 --hidden1 76 --keep_prob0 0.5 --use_elmo --pre_att_id SKIP-v16-cnn-gru-yann-ly3-k4q4v1-lr8.0-sp4-topk1.0-declen3 --original_ptr --condition 1 --gate_fuse 3 --att_cnt 2
`

`
python main.py --mode train --no_bucket --keep_prob 0.8 --ptr_keep_prob 0.8 --batch_size 32 --hidden0 76 --hidden1 76 --keep_prob0 0.5 --use_elmo --pre_att_id SKIP-v16-cnn-gru-yann-ly3-k4q4v1-lr8.0-sp4-topk1.0-declen3 --original_ptr --condition 1 --gate_fuse 3 --att_cnt 2 --trnn
`



