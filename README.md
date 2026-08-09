# Timbre interpolation with Differentiable Digital Signal Processing

This code started as a fork of the implementation of the [DDSP model](https://github.com/magenta/ddsp) in PyTorch [made by ACIDS-IRCAM](https://github.com/acids-ircam/ddsp_pytorch). It was modified to use the [UMRP Dataset](https://labsites.rochester.edu/air/projects/URMP.html), which consists of several subfolders with different instruments. This structure of folders is assumed for preprocessing and training.

This repository is tailored for offline interpolation between instruments, and realtime considerations were left for future work.

Evaluation is done using the code from [BRAVE](https://github.com/fcaspe/BRAVE)

## Setup

Create and activate a virtual environment, then install the dependencies:

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Structure and further instructions
The structure is divided into three blocks: **Train, Inference and Evaluation.** 
For detailed instructions on each block, see their README files:

* [Train](https://github.com/jimearruti/ddsp_interpolate/blob/main/train/README.md)
* [Inference](https://github.com/jimearruti/ddsp_interpolate/blob/main/inference/README.md)
* [Evaluation](https://github.com/jimearruti/ddsp_interpolate/blob/main/evaluation/README.md)

Remember that everything is run from the root folder!



