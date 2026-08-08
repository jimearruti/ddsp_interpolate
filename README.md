# Timbre interpolation with Differentiable Digital Signal Processing

This code is based on the implementation of the [DDSP model](https://github.com/magenta/ddsp) in PyTorch [made by ACIDS-IRCAM](https://github.com/acids-ircam/ddsp_pytorch).

The original code was modified to use the [UMRP Dataset](https://labsites.rochester.edu/air/projects/URMP.html), which consists of several subfolders with different instruments. This structure of folders is assumed for preprocessing and training.

This repository is tailored for offline interpolation between instruments, and realtime considerations were left for future work.

The structure is divided into three blocks: **Train, Inference and Evaluation.** 
For detailed instructions on each block, see their README files:

* [Train](https://github.com/jimearruti/ddsp_interpolate/blob/main/train/README.md)
* [Inference](https://github.com/jimearruti/ddsp_interpolate/blob/main/inference/README.md)
* [Evaluation](TODO)

Remember that everything is run from the root folder!

## Main changes from original repo
* Added data split logic
* Automated training of separate models for each instrument
* Added base model and finetuning logic
* Implemented exponential decay for training as described in the DDSP paper
* Corrected issues pointed in reference repo (Loudness, MLP bug)
* Corrected std calculation
* Everything related to interpolation





