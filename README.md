# Timbre interpolation with Differentiable Digital Signal Processing

## Structure and further instructions
The structure of the repo is divided into three blocks: **Train, Inference and Evaluation.** 
For detailed instructions on each block, see their README files:

* [Train](https://github.com/jimearruti/ddsp_interpolate/blob/main/train/README.md)
* [Inference](https://github.com/jimearruti/ddsp_interpolate/blob/main/inference/README.md)
* [Evaluation](https://github.com/jimearruti/ddsp_interpolate/blob/main/evaluation/README.md)

Remember that everything is run from the root folder!

> Note that the **interpolation of parameters** is called **interpolation of outputs** in this code.

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

## Acknowledgments
This code started as a fork of the implementation of the [DDSP model](https://github.com/magenta/ddsp) in PyTorch [made by ACIDS-IRCAM](https://github.com/acids-ircam/ddsp_pytorch). 

The [UMRP Dataset](https://labsites.rochester.edu/air/projects/URMP.html) was used for the experiments.

Evaluation is based on code from [BRAVE](https://github.com/fcaspe/BRAVE)


