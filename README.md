# Interpretable Adversarial Perturbation
Code for [*Interpretable Adversarial Perturbation in Input Embedding Space for Text*](https://arxiv.org/abs/1805.02917), IJCAI 2018.

This code reproduce the our paper with [Chainer](https://github.com/chainer/chainer).


## Setup Environment
Please install [Chainer](https://github.com/chainer/chainer) and [Cupy](https://github.com/cupy/cupy).

You can set up the environment easily with this [*Setup.md*](https://github.com/aonotas/interpretable-adv/blob/master/Setup.md).

## Download Pretrain Model
Please download pre-trained model. Note that this pretrained model is genera
```
$ wget http://sato-motoki.com/research/vat/imdb_pretrained_lm_ijcai.model
```


# Run
## Pretrain
```
$ python -u pretrain.py -g 0 --layer 1 --dataset imdb --bproplen 100 --batchsize 32 --out results_imdb_adaptive --adaptive-softmax
```
Note that this command takes about 30 hours with single GPU.

## Train (iVAT: Interpretable Semi-supervised setting)
Please add `--use_semi_data 1` and `--use_attn_d 1` to use iVAT (ours).
```
$ python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_vat --lower=0 --use_adv=0 --xi_var=5.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm_ijcai.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1 --use_attn_d 1
```

## Train (VAT: Semi-supervised setting)
Please add `--use_semi_data 1` to use VAT.
```
$ python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_vat --lower=0 --use_adv=0 --xi_var=5.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm_ijcai.model --use_exp_decay=1 --clip=5.0 --batchsize_semi 96 --use_semi_data 1
```
Note that this command takes about 8 hours with single GPU.

## Train (iAdv: Interpretable Supervised setting)
Please add `--use_adv 1` and `--use_attn_d 1` to use iAdv.
```
$ python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_adv --lower=0 --use_adv 1 --xi_var=5.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm_ijcai.model --use_exp_decay=1 --clip=5.0
```

## Train (Adv: Supervised setting)
Please add `--use_adv 1` to use Adv.
```
$ python train.py --gpu=0 --n_epoch=30 --batchsize 32 --save_name=imdb_model_adv --lower=0 --use_adv 1 --xi_var=5.0  --use_unlabled=1 --alpha=0.001 --alpha_decay=0.9998 --min_count=1 --ignore_unk=1 --pretrained_model imdb_pretrained_lm_ijcai.model --use_exp_decay=1 --clip=5.0
```
Note that this command takes about 6 hours with single GPU.

# Authors
We thank Takeru Miyato ([@takerum](https://github.com/takerum)) who suggested that we reproduce the result of a [Miyato et al., 2017].
- Code author: [@aonotas](https://github.com/aonotas/)
- Thanks for Adaptive Softmax implementation: [@soskek](https://github.com/soskek/)
Adaptive Softmax: https://github.com/soskek/efficient_softmax
# Reference
```
[Miyato et al., 2017]: Takeru Miyato, Andrew M. Dai and Ian Goodfellow
Adversarial Training Methods for Semi-Supervised Text Classification.
International Conference on Learning Representation (ICLR), 2017

[Sato et al., 2018]: Motoki Sato, Jun Suzuki, Hiroyuki Shindo, Yuji Matsumoto
Interpretable Adversarial Perturbation in Input Embedding Space for Text.
IJCAI-ECAI-2018
```
# TODO
- Add visualizing code
