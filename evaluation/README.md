# Evaluation

## Postprocess original recordings

In the preprocessing stage the input signals are normalised to -1dB and filtered with a high pass filter with a cutoff frequency at 80Hz. This preprocessing, however, is done and reflected in the data stored for training, but not the original audio is not affected. To obtain the original recordings as they are seen by the model, run

```
python -m evaluation.postprocess_original_recordings
```

The normalised and filtered audio will be stored by default in `<original_dataset_root_path>/normalised`


## Reorder dataset
Since the metric requires to group audio in folders, the dataset can be reordered with:

```
python -m evaluation.reorder_dataset <split_json> <normalised_dataset_source_root> <dest_root>
```

This will reorganise the dataset as needed, generating folders per split and instrument instead of dividing by pieces. Results will be saved on `<dest_root>` accordingly,

## Reorder results
The dataset can be reordered with:

```
python -m evaluation.reorder_results <src_results_dir> <dst_results_dir>
```
This will reorganise the results, generating folders for each type of result in `dst_results_dir`


## Frechet Audio Distance

To calculate the Frechet Audio Distance between two groups of audio, a JSON file needs to be created defining what the `background_path` is, that is, the one that has the recordings we will compare with, and what the different `resynth_paths` are. One metric will be computed for each one.

For example:

```
{
    "background_path": "../data/URMP/Dataset/rearranged/train_vn",
    "resynth_paths": [
        "../data/URMP/Dataset/rearranged/test_vn",
    ]
}
```

Then run

```
python -m evaluation.fad <config.json>
```




