# Inference Block

## Adjust configuration
After you have trained at least one model per instrument, it's time to generate results.

First, open `generate_config.yaml` and adjust the type of models to use and their paths.
```
  model_types:
    - finetuned_from_30k_for_30k
    - finetuned_from_30k_for_15k
    - finetuned_from_15k_for_15k
    - finetuned_from_15k_for_30k
```
```
instrument_paths:
  vn:
    from_scratch:               runs/normalised_high_pass_from_scratch/20260728_102000/vn/state_15000.pth
    finetuned_from_30k_for_30k: runs/normalised_high_pass_filter_finetuned/20260727_230936/vn/state_30000.pth
    finetuned_from_30k_for_15k: runs/normalised_high_pass_filter_finetuned/20260727_230936/vn/state_15000.pth
    finetuned_from_15k_for_30k: runs/normalised_high_pass_finetuned_from_15k/20260728_162012/vn/state_30000.pth
    finetuned_from_15k_for_15k: runs/normalised_high_pass_finetuned_from_15k/20260728_162012/vn/state_15000.pth
  fl:
    from_scratch:               runs/normalised_high_pass_from_scratch/20260728_102000/fl/state_15000.pth
    finetuned_from_30k_for_30k: runs/normalised_high_pass_filter_finetuned/20260727_230936/fl/state_30000.pth
    finetuned_from_30k_for_15k: runs/normalised_high_pass_filter_finetuned/20260727_230936/fl/state_15000.pth
    finetuned_from_15k_for_30k: runs/normalised_high_pass_finetuned_from_15k/20260728_162012/fl/state_30000.pth
    finetuned_from_15k_for_15k: runs/normalised_high_pass_finetuned_from_15k/20260728_162012/fl/state_15000.pth
  tpt:
    from_scratch:               runs/normalised_high_pass_from_scratch/20260728_102000/tpt/state_15000.pth
    finetuned_from_30k_for_30k: runs/normalised_high_pass_filter_finetuned/20260727_230936/tpt/state_30000.pth
    finetuned_from_30k_for_15k: runs/normalised_high_pass_filter_finetuned/20260727_230936/tpt/state_15000.pth
    finetuned_from_15k_for_30k: runs/normalised_high_pass_finetuned_from_15k/20260728_162012/tpt/state_30000.pth
    finetuned_from_15k_for_15k: runs/normalised_high_pass_finetuned_from_15k/20260728_162012/tpt/state_15000.pth
```

Adjust the desired values for alpha and where you want the results to be saved
 ```
results_dir: results
 alphas: [0.0, 0.25, 0.5, 0.75, 1.0]
```

## Generate outputs
Run the following code to automatically generate results
```
python -m inference.generate_audio
```

For each instrument pair and each type of model, the following will be generated:
TODO

## Normalise results

## Crop results

