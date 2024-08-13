# JEPAs
Un-official PyTorch implementations of:
- [*] I-JEPA: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [*] V-JEPA: [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471)
- [ ] MC-JEPA: [MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features](https://arxiv.org/abs/2307.12698)
- [ ] Graph-JEPA: [Graph-level Representation Learning with Joint-Embedding Predictive Architectures](https://arxiv.org/abs/2309.16014)

## I-JEPA Explained
(JEPA is just an architectural and optimisational "slight of hand" for pretraining Vision Transformers that enables them to learn expressive features by predicting the latent space representations of uncorrupted portions of images given image inputs subject to occlusions.)

I-JEPA exploits the following truth:
The latent space representations is all you need for understanding.

Picture your mother - one of the faces you are most familiar with.
How detailed is that picture, however?
Not very - because it doesn't need to be.

You likely have learned highly expressive latent representations for things you understand best/are most familiar with.
You do not need to be able to perform high resolution reconstruction (ViT MAE), however, to achieve expert understansing.
You just need a good latent representation.

How do we train a good latent representation?
Train a model to predict how it would represent patches of an image *if it could see them*.
In a self-supervised manner, we can encode patches with a target encoder (teacher), that are masked to a context encoder (student), and task the context encoder with predicting the encoding produced by the target encoder.
By constructing an appropriate loss function, we can train the student encoder to match the teacher encoder, as the teacher encoder learns better representations.
The only way for the student encoder to predict the teacher encoder's patches is to learn robust representations in a vision "world model".

Basic Schematic of Architecture:  
![screenshot](IJEPA.png)

## Usage:
### I-JEPA
Run a pretraining job with the [imagenet](https://www.image-net.org/) dataset (stored locally at [relative/path/to/splits](data/imagenet/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC)):
```bash
python pretrain_IJEPA.py
```

# TODO
Run a finetuning job...
```bash
python finetune_IJEPA.py
```

## TODO:
- [ ] Linear probing setup

## Acknowledgements
- The above implementations use [@lucidrains](https://github.com/lucidrains) x-transfromers (https://github.com/lucidrains/x-transformers)

## Citation:
```
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={arXiv preprint arXiv:2301.08243},
  year={2023}
}
```
