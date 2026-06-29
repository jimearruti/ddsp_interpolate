# Timbre interpolation with Differentiable Digital Signal Processing

This code is based on the implementation of the [DDSP model](https://github.com/magenta/ddsp) in PyTorch [made by ACIDS-IRCAM](https://github.com/acids-ircam/ddsp_pytorch).

The original code was modified to use the [UMRP Dataset](https://labsites.rochester.edu/air/projects/URMP.html), which consists of several subfolders with different instruments. This structure of folders is assumed for preprocessing and training.

This repository is tailored for offline interpolation between instruments, and realtime considerations were left for future work.

## Training from scratch

Edit the `config.yaml` file to fit your needs (instruments to consider, audio location, preprocess folder, sampling rate, model parameters...), then preprocess your data using 

```bash
python preprocess.py
```

This will process each instrument separately.

You can then train a model for each instrument by calling 

```bash
python train.py --name mytraining (--steps 10000000 --batch 16 --lr .001 --instrument vn)
```

Once models are trained, they can be exported by using

```bash
python export.py --run runs/mytraining/
```

It will produce a file named `ddsp_pretrained_mytraining.ts`, that can later be used to produce sound.



> By default, preprocessing and training were modified to perform a split of the dataset, leaving a holdout test set. Models can be trained with all the dataset by adding `--split_dataset false` to the preprocessing and training calls.

## Finetuning from a base model

### Creating a base model
> By default, we assume the base model will be trained on all the instruments except violin, trumpet and saxophone (since those will be used for finetuning and testing purposes). If another behaviour is intended, the following mentioned Python scripts should be modified.

To obtain a base model, you need to run
```bash
python preprocess_base.py
```
Later, run

```bash
python train_base_model.py
```

This will create a base model using data from several instruments. Audio quality is not expected to be good, since it will serve as a starting point for finetuning.

### Finetuning
After the base model is obtained, a normal preprocess and train cycle as described above (in training from scratch) should be run, adding `--base_model_path /path/to/base/model` to the call.

```bash
python train.py --name mytraining --base_model_path /path/to/base/model (--steps 10000000 --batch 16 --lr .001 --instrument vn)
```

## Playing with models
TO DO

## Main changes from original repo
* Automate training of separate models for each instrument
* Add base model and finetuning logic
* Implement exponential decay for training as described in the DDSP paper
* Corrected issues pointed in reference repo






