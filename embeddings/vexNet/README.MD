# Training the VexIR2Vec Model

To train VexIR2Vec, two files in `.h5` format are needed:
1. **Keys File**
2. **Training Data File**

These can be generated using the following scripts:

###  Generating Keys File
[`gen_keys_h5.py`](./gen_keys_h5.py) generates the Keys `.h5` file.
```bash
  python gen_keys_h5.py <space-separated list of project names> ...
```

### Generate Training Data File
[`gen_training_data_from_csvs.py`](./gen_training_data_from_csvs.py) generates the
training data `.h5` file used for VexIR2Vec training.
```bash
python gen_training_data_from_csvs.py <space-separated list of project names>
```

### Training the VexNet Model

```
python vexir2vec_training.py -opt <optimizer> --beta <value> --lr <learning-rate> --gamma <value> --batch_size <batch size> -e <no-of-epochs> -temp <value> --inp_dim <input_dim> --out_dim <output-dim> --best_model_path models/vexir2vec_model/
```

> Key Parameters for `vexir2vec_training.py`

| Parameter       | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| `-bs`           | Batch size (e.g., `1024`)                                                    |
| `-e`            | Number of epochs (e.g., `150`)                                               |
| `-temp`            | temperature for NTXent loss (e.g., `0.8`)                                        |
| `-inpd`         | Input dimension (e.g., `144`)                                                |
| `-outd`         | Output embedding dimension (e.g., `100`)                                     |
| `-opt`          | Optimizer to use (e.g., `adam`)                                              |
| `-lr`           | Learning rate (e.g., `0.005`)                                                |
| `-bmp`          | Base model path (where to save the trained model)                           |
| `-tune`         | Usage of Ray (true/false)                                                   |
| `-beta`         | Beta value                                                                  |


The following config defines Additional training configuration parameters used in [`vexir2vec_training.py`](./vexir2vec_training.py#L550):

```
config = {
            "activation": "<str: activation function>",  # e.g., 'relu', 'leaky_relu', 'tanh', 'gelu'
            "batch_size": args.batch_size,  # batch size, typically power of 2
            "beta": args.beta,  # e.g., beta1/beta2 for Adam optimizer or loss scaling
            "concat_layer": "<int: 0-num_layers-1>",  # index at which to apply layer concatenation
            "drop_units": "<list[float]: each in 0.0-0.5>",  # dropout rate per layer
            "gamma":  "<float: 0.0-1.0>",  # scheduler decay factor or discount rate
            "hidden": "<list[int]: e.g., [64-1024, 64-1024]>",  # hidden units per layer
            "lr": args.lr,  # learning rate
            "temperature": args.temperature,
            "num_O_layers":"<int: 1-3>",  # number of output layers
            "num_layers": "<int: 1-5>",  # number of hidden layers
            "opt": args.optimizer,  # e.g., 'adam', 'sgd', 'adamw'
            "sched": "Linear_lr",  # e.g., 'Linear_lr', 'StepLR', 'CosineAnnealing'
            "thresh_max": "<int: > thresh_min>",  # upper threshold value
            "thresh_min": "<int: < thresh_max>",  # lower threshold value
        }
```

- Parameters such as batch_size, lr, beta, temperature, and opt—which have a significant impact on training performance—are passed via argparse. The remaining parameters should be manually configured within the vexir2vec_training.py script.
