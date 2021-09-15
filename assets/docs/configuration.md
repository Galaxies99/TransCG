# Configurations

Configuration files should be created for training, testing and inference. Our configuration file has a `yaml` format. The following items and sub-items are provided in configuration file.

```yaml
"model":
  "type": # str (default: 'DFNet'), model type, now only support "DFNet".
  "params": # parameters for the model.
    "in_channels": # int (default: 4), the input channels of the model (default: 4).
    "hidden_channels": # int (default: 64), the hiden channels of the model.
    "L": # int (default: 5), the number of layers in a dense block.
    "k": # int (default: 12), the output channel of a single layer in a dense block.
    "use_DUC": # bool (default: True), whether to use the DUC instead of the deconvolution in up-sampling process.

"optimizer":
  "type": # str (default: 'AdamW'), optimizer type, refer to objects in torch.optim.
  "params": # parameters for the optimizer, see specific optimizer parameters in torch for details.

"lr_scheduler":
  "type": # str (default: ''), learning rate scheduler type, refer to objects in torch.optim.lr_scheduler. Specially, '' for no learning rate scheduler.
  "params": # parameters for the learning rate scheduler, see specific learning rate scheduler parameters in torch for details.

"dataset":
  "train": # list or dict, one item for one dataset.
    "type": # str (default: 'transcg'), dataset type, support type: ['transcg', 'cleargrasp-real', 'cleargrasp-syn', 'omniverse', 'transparent-object']
    "data_dir": # str, directory to the data
    "image_size": # tuple of (int, int) (default: (1280, 720)), the resolution of the data (samples will be scaled to this size and fed into the network).
    "use_augmentation": # bool (default: True), whether to use data augmentation.
    "rgb_augmentation_probability": # float (default: 0.8), RGB augmentation probability.
    "depth_min": # float (default: 0.3), the minimum depth.
    "depth_max": # float (default: 1.5), the maximum depth
    "depth_norm": # float (default: 1.0), the depth normalization coefficient.
  "test": # list or dict, one item for one dataset.
    # The dataset description is the same as the training dataset description.

"dataloader": 
  # the following parameters are the same as parameters in torch.utils.data.DataLoader.
  "num_workers": # the number of workers in data loader.
  "shuffle": # whether shuffle the data.
  "drop_last": # whether drop the last data (that is not in a batch).

"trainer": 
  "batch_size": # int (default: 32), batch size
  "test_batch_size": # int (default: 1), batch size during testing.
  "multigpu": # bool (default: False), whether to use the multigpu for training/testing.
  "max_epoch": # int (default: 40), the number of epochs.
  "criterion":
    "type": # main loss type, support type: ['mse_loss', 'masked_mse_loss', 'custom_masked_mse_loss', 'l1_loss', 'masked_l1_loss', 'custom_masked_l1_loss', 'l2_loss', 'masked_l2_loss', 'custom_masked_l2_loss','huber_loss', 'masked_huber_loss', 'custom_masked_huber_loss']
    "epsilon": # float (default: 1e-8), the epsilon to avoid NaN during calculation.
    "huber_k": # float (default: 0.1) the k value in huber loss. The parameter is only used when using loss type concerning huber loss.
    "combined_smooth": # bool (default: False), whether to combine the smooth loss calculated by the cosine similarity between surface normals of ground-truth depth maps and predicted depth maps.
    "combined_beta": # float (default: 0.005), the balance coefficient between depth loss and the smooth loss.
    "combined_beta_decay": # float (default: 0.1), the balance coefficient decay value.
    "combined_beta_decay_milestones": # list (default: []), the milestones of balance coefficient decay.

"metrics":
  "types": # list of str (default: []), metric types, supporting type: ["MSE", "MaskedMSE", "RMSE", "MaskedRMSE", "REL", "MaskedREL", "MAE", "MaskedMAE", "Threshold@k", "MaskedThreshold@k"] (where k is a float value denoting "delta" in the paper).
  "epsilon": # float (default: 1e-8), the epsilon to avoid NaN during calculation.
  "depth_scale": # float (default: 1.0), the depth scale, should be the same as "depth_norm" in the testing set.

"stats":
  "stats_dir": # the directory to the statistics.
  "stats_exper": # the experiment directory to the statistics.
```
