# TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and Grasping

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transcg-a-large-scale-real-world-dataset-for/transparent-object-depth-estimation-on)](https://paperswithcode.com/sota/transparent-object-depth-estimation-on?p=transcg-a-large-scale-real-world-dataset-for) [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[[Paper]](https://arxiv.org/pdf/2202.08471)  [[Project Page]](https://graspnet.net/transcg)

**Authors**: [Hongjie Fang](https://github.com/galaxies99/), [Hao-Shu Fang](https://github.com/fang-haoshu), [Sheng Xu](https://github.com/XS1020), [Cewu Lu](https://mvig.sjtu.edu.cn/).

Welcome to the official repository for the TransCG paper. This repository includes the dataset and the proposed Depth Filler Net (DFNet) models.

## TransCG Dataset

<img align="right" src="assets/imgs/TransCG.gif" width=240px> TransCG dataset is now available on [official page](https://graspnet.net/transcg). TransCG dataset is the first large-scale real-world dataset for transparent object depth completion and grasping. In total, our dataset contains 57,715 RGB-D images of 51 transparent objects and many opaque objects captured from different perspectives of 130 scenes under various real-world settings. The 3D mesh model of the transparent objects are also provided in our dataset.

<table>
  <tr><td><img src='assets/imgs/object.png' width=320px></td><td><img src='assets/imgs/tracking-system.gif' width = 256px></td><td><img src='assets/imgs/robot-collection.gif' width=256px ></td></tr>
  <tr><td align="center"> Daily Transparent Objects in Dataset</td><td align="center"> Real-time Tracking System</td><td align="center">Robot Collection</td></tr>
</table>

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

Our pretrained checkpoint is available on [Google Drive](https://drive.google.com/file/d/1APIuzIQmFucDP4RcmiNV-NEsQKqN9J57/view?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/14khejj63OjOKsyzxnuYo5Q) (Code: c01g). The checkpoint is trained with the default configuration in the `configs` folder. You can use our released checkpoints for [inference](#inference) or [testing](#testing-optional). Refer to [assets/docs/DFNet.md](assets/docs/DFNet.md) for details about the depth completion network.

### Grasping Demo

To verify the depth completion results in robotic manipulation, we select the fundamental object grasping as the downstream task of our network. Here are the grasping demos. Refer to [assets/docs/grasping.md](assets/docs/grasping.md) for details about the grasping method.

<table>
  <tr><td><img src='assets/imgs/grasp-1.gif' width=256px></td><td><img src='assets/imgs/grasp-2.gif' width = 256px></td><td><img src='assets/imgs/grasp-3.gif' width=256px ></td></tr>
</table>

### Configuration

You need to create a configuration file for training, testing and inference. See [assets/docs/configuration.md](assets/docs/configuration.md) for details.

### Dataset Preparation

- **TransCG** (recommended): See [TransCG Dataset](#transcg-dataset) section;
- **ClearGrasp** (syn and real): See [ClearGrasp official page](https://sites.google.com/view/cleargrasp);
- **Omniverse Object Dataset**: See [implicit-depth official repository](https://github.com/NVlabs/implicit_depth);
- **Transparent Object Dataset**: See [KeyPose official page](https://sites.google.com/view/keypose).

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

### Training (Optional)

For training from scrach, you need to create a configuration file following instruction of [configuration section](#configuration). Then, execute the following commands to train your own model.

```bash
python train.py --cfg [Configuration File]
```

If you want to fine-tune your model from some checkpoints, you may need to provide `resume_lr` in configuration file. See [assets/docs/configuration.md](assets/docs/configuration.md) for details.

### Testing (Optional)

For model testing, you also need to create a configuration file following instruction of [configuration section](#configuration). Then, execute the following commands to test the model.

```bash
python test.py --cfg [Configuration File]
```

**Note**. For testing stage, the checkpoint specified in the configuration file should exist.

## Citation

```bibtex
@article{fang2022transcg,
    title   = {TransCG: A Large-Scale Real-World Dataset for Transparent Object Depth Completion and Grasping},
    author  = {Fang, Hongjie and Fang, Hao-Shu and Xu, Sheng and Lu, Cewu},
    journal = {arXiv preprint arXiv:2202.08471}
    year    = {2022}
}
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg