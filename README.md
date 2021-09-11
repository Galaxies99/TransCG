# Depth Filler Net (DFNet)

The official repository for **Transparent-Grasp: A Large-Scale Real-World Depth Completion Dataset for Transparent Object Grasping**.

## Requirements

The code has been tested under

- Ubuntu 18.04 + NVIDIA GeForce RTX 3090 (CUDA 11.1)
- PyTorch 1.9.0

Other dependencies can be installed by

```bash
pip install -r requirements.txt
```

## Run

### Quick Start

Our pretrained checkpoints and configuration files are available here.

### Configuration

### Training

```bash
python train.py --cfg [Configuration File]
```

### Testing

```bash
python test.py --cfg [Configuration File]
```

### Inference

For inference stage, there is a `Inferencer` class in `inference.py`, you can directly call it for inference. See `sample_inference.py` for details.
