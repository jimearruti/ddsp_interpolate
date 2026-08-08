## Train block

### Dataset split

Edit `../config.yaml` to adjust ratios for validation and train splits, and insert the preprocess folder you want the data to be split in. Defaults are:
```
val_ratio: 0
test_ratio: 0.3
```

After, while in the root directory, perform the data split using

```bash
python -m train.split_dataset
```

The split is done on a **piece** level, to ensure no leakage takes place. There's a check on a file level after the split is done to ensure the ratios are approximately met for **violin, flute and trumpet**.

The split used for our experiments can be seen in `preprocessed/split_files.json`.

### Data preprocessing

Edit `../config.yaml` file to fit your needs (instruments to consider, audio location, preprocess folder, sampling rate, model parameters...), then preprocess your data using 

```bash
python -m train.preprocess
```

This will process each instrument separately.


### Train a model from scratch
You can then train a model for each instrument by calling 

```bash
python -m train.train --name mytraining_from_scratch (--steps 30000 --instrument vn)
```
If no instrument is specified, a model for each instrument defined in `config.yaml` will be trained.

### Finetuning from a base model

#### Creating a base model

To obtain a base model, you need to run

```bash
python -m train.train_base_model --name mytraining_base_model
```

This will create a base model using data from all instruments defined in `config.yaml`. Audio quality is not expected to be good, since it mixes recordings of all the instruments and will serve as a starting point for finetuning. 

The trained base models are saved in `runs/<training_name>/<timestamp_with_date_and_time_info>/base_model/state_<checkpoint_number>.pth`

#### Finetuning
After the base model is obtained, a normal train cycle as described above (in training from scratch) should be run, adding `--base_model_path /path/to/base/model` to the call. 

```bash
python -m train.train --name mytraining_finetuning --base_model_path /path/to/base/model (--steps 30000 --instrument vn)
```

