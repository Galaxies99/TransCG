# TransCG: A Large-Scale Real-World Dataset for Transparent ObjectDepth Completion and Grasping

**Authors**: [Hongjie Fang](https://github.com/galaxies99/), [Hao-Shu Fang](https://github.com/fang-haoshu), [Sheng Xu](https://github.com/XS1020), [Cewu Lu](https://mvig.sjtu.edu.cn/).

Welcome to the official repository for the TransCG paper. This repository includes the dataset and the proposed Depth Filler Net (DFNet) models.

## TransCG Dataset

TransCG dataset is available on [Google Drive](link) and [Baidu Netdisk](link).

## Requirements

The code has been tested under

- Ubuntu 18.04 + NVIDIA GeForce RTX 3090 (CUDA 11.1)
- PyTorch 1.9.0

System dependencies can be installed by:

```bash
sudo apt-get install libhdf5-10 libhdf5-serial-dev libhdf5-dev libhdf5-cpp-11
sudo apt install libopenexr-dev zlib1g-dev openexr
```

Other dependencies can be installed by

```bash
pip install -r requirements.txt
```

## Run

### Quick Start

Our pretrained checkpoints and configuration files are available [here](link).

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

For inference stage, there is a `Inferencer` class in `inference.py`, you can directly call it for inference. 

**Example**. Given an `H x W x 3` RGB image `rgb`, and an `H x W` depth image `depth` (after scaling according to camera parameters), you can use the following code to get the refined depth according to our models.

```python
from inferencer import Inferencer
# Initialize the inferencer. It is recommended to intiailize before starting your task for real-time performance.
inferencer = Inferencer(cfg_file = 'configs/inference.yaml') # Specify your configuration file here.
# Call inferencer for refined depth
refine_depth = inferencer.inference(rgb, depth)
```

For full code sample, refer to `sample_inference.py`.