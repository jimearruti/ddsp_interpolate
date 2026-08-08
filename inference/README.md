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

Adjust the desired values for alpha and where you want the results to be saved (default is `results`)
 ```
results_dir: results
 alphas: [0.0, 0.25, 0.5, 0.75, 1.0]
```

## Generate outputs
Run the following code to automatically generate results
```
python -m inference.generate_audio
```

A folder is generated for each piece in the test dataset played by a violin, flute or trumpet. Results corresponding to that recording are stored in that folder.

For each instrument pair (instrument1 -> instrument2) and each type of model, the following will be generated:
* output for the instrument1 model: `<input_name>_<instrument1>_<model_type>.wav`
* output for the instrument2 model: `<input_name>_<instrument2>_<model_type>.wav`
* output obtained by interpolating the synth parameters calculated by the models for the chosen alphas: `<input_name>_interpolated_outputs_<instrument1>_<instrument2>_<model_type>_alpha_<alpha>_<with/without>_reverb.wav`
* output obtained by interpolating the weights of the models for the chosen alphas: `<input_name>_interpolated_weights_<instrument1>_<instrument2>_<model_type>_alpha_<alpha>_<with/without>_reverb.wav`
* output obtained by interpolating the synth parameters with a swept alpha: `<input_name>_sweep_output_<instrument1>_<instrument2>_<model_type>_alpha_<alpha>_<with/without>_reverb.wav`
* output obtained by interpolating the weights of the models with a swept alpha:
`<input_name>_sweep_output_<instrument1>_<instrument2>_<model_type>_alpha_<alpha>_<with/without>_reverb.wav`

## Normalise results
If you want to compare results, it's useful to normalise to the same loudness.

Adjust the value of LUFS as desired in `generate_config.yaml`. Default is -24 LUFS.

Results are saved in `results_normalised`

```
python -m inference.postprocess
```

## Crop results
Assuming you want to select parts of the pieces to compare results in a listening test, adjust the following for each piece in `generate_config.yaml`

```
  crop:
    pieces:
      AuSep_1_vn_02_Sonata:
        original_path: ../data/URMP/normalised/02_Sonata_vn_vn/AuSep_1_vn_02_Sonata.wav
        start_time: 4.330
        end_time: 12.569
```

Then run the following to automatically generate cropped versions of the results and original recording, normalised in loudness to -24 LUFS.

```
python -m inference.crop_outputs
```
The cropped results are saved in `results_cropped` adding timestamps for the start and end of the crop in the filename.

