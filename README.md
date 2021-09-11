# TransCG: A Large-Scale Real-World Dataset for Transparent ObjectDepth Completion and Grasping

**Authors**: [Hongjie Fang](https://github.com/galaxies99/), [Hao-Shu Fang](https://github.com/fang-haoshu), [Sheng Xu](https://github.com/XS1020), [Cewu Lu](https://mvig.sjtu.edu.cn/).

Welcome to the official repository for the TransCG paper. This repository includes the dataset and the proposed Depth Filler Net (DFNet) models.

## TransCG Dataset

TransCG dataset is available on [Google Drive](link) and [Baidu Netdisk](link).

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

You need to create a configuration file for training, testing and inference. See [docs/configuration](docs/configuration.md) for details.

### Dataset Preparation

- **TransCG** (recommended): See [TransCG Dataset](#transcg-dataset) section;
- **ClearGrasp** (syn and real): See [ClearGrasp official page](https://sites.google.com/view/cleargrasp);
- **Omniverse Object Dataset**: See [implicit-depth official repository](https://github.com/NVlabs/implicit_depth);
- **Transparent Object Dataset**: See [KeyPose official page](https://sites.google.com/view/keypose).

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
