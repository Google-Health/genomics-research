# REGLE (REpresentation learning for Genetic discovery on Low-dimensional Embeddings)

This repository contains code for defining and training models for spirogram
encodings ("SPINCs") and plethysmogram (PPG) encodings ("PLENCs"), using the
REGLE ("REpresentation learning for Genetic discovery on Low-dimensional
Embeddings") framework described in this manuscript:
"[Unsupervised representation learning improves genomic discovery for respiratory and circulatory functions and diseases](https://doi.org/10.1101/2023.04.28.23289285)".
It also includes the trained model checkpoints used for the experiments in the
manuscript.

Please cite the above manuscript if you use this repository for your research.


## Running demo using the trained model

The following steps will generate a demo dataset consisting of a single
spirogram and generate (R)SPINCs of that spirogram using the trained VAE models.
We use a publicly available example that can be downloaded from the UK Biobank
webpage: https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=3

First, install required packages. (We recommend using a virtual environment.)

```
$ pip3 install -r requirements.txt
```

Second, generate a demo dataset in `demo` directory.

```
$ mkdir demo
$ python3 generate_ukb_3066_demo_dataset.py --out_dir=demo
```

Third, generate SPINCs (dim=5) using the demo dataset and print output:

```
$ python3 generate_spincs.py \
  --input_path=demo/ukb_3066_demo.flow_volume_in_channels.npy \
  --output_path=demo/ukb_3066_demo.spincs.npy \
  --model_type=spincs \
  --print_output
```

The expected output values are
`-1.2216355, -0.2726544, 2.2720308, -0.7533066, -0.8892823`
or something similar.

Finally, generate RSPINCS (dim=2) using the demo dataset and print output:

```
$ python3 generate_spincs.py \
  --input_path=demo/ukb_3066_demo.flow_by_volume_one_channel.npy \
  --output_path=demo/ukb_3066_demo.rspincs.npy \
  --model_type=rspincs \
  --print_output
```

The expected output values are
`-0.7540935, -1.8171966`
or something similar.


## Training a SPINCs or RSPINCs model using your data

This time we will generate two sets of demo data, training and validation,
by duplicating the demo data multiple times. In practice, you can use your
own spirograms to train the (R)SPINCs model by following the same steps.

```
$ mkdir demo_train
$ mkdir demo_val
$ python3 generate_ukb_3066_demo_dataset.py \
  --out_dir=demo_train \
  --duplicates=80
$ python3 generate_ukb_3066_demo_dataset.py \
  --out_dir=demo_val \
  --duplicates=20
```

Now we are ready to train (R)SPINCs model using the demo datasets.

### Training a SPINCs model

We use `train_model.py` to train a VAE model for the given spirograms.
The most important argument is `latent_dim`, which determines the dimension
of the latent variables (i.e. SPINCs). We used 5 dimensions in the paper, but
you are free to choose other values.
Here we train it for just 1 epoch with a fixed set of hyperparameters using
Adam optimizer. You can additionally pass your own hyperparameters to
other arguments, such as `random_seed`, `learning_rate`,
`batch_size`, and `num_epochs`.

```
$ python3 train_model.py \
  --input_train=demo_train/ukb_3066_demo.flow_volume_in_channels.npy \
  --input_validation=demo_val/ukb_3066_demo.flow_volume_in_channels.npy \
  --latent_dim=5 \
  --output_dir=demo_spincs_train_output
```

You can inspect the training history in the generated JSON file:

```
$ cat demo_spincs_train_output/training_history.json
```

The trained Keras model with the best validation loss is saved in
`checkpoint/best-cp.ckpt` in the output dir.

### Training a RSPINCs model with expert-defined features (EDFs)

`train_model.py` can be used to train a RSPINCs model using the `--rspincs`
flag, which accepts additional expert-defined features (EDFs).
In the paper we used 5 EDFs, namely FEV1, FVC, PEF, FEV1/FVC ratio, and
FEF25-75%, and used 2 latent dimensions for RSPINCs.
For RSPINCs, `edf_dim` argument must be specified,
and the numpy array of EDFs is assumed to have shape:
(num_individuals, edf_dim).
When using your own data, we recommend standardizing each EDF to have zero mean
and unit variance before training a RSPINCs model.

```
$ python3 train_model.py \
  --input_train=demo_train/ukb_3066_demo.flow_by_volume_one_channel.npy \
  --input_train_edfs=demo_train/ukb_3066_demo.derived_features.npy \
  --input_validation=demo_val/ukb_3066_demo.flow_by_volume_one_channel.npy \
  --input_validation_edfs=demo_val/ukb_3066_demo.derived_features.npy \
  --latent_dim=2 \
  --edf_dim=5 \
  --output_dir=demo_rspincs_train_output \
  --rspincs
```

Again, you can inspect the training history in the generated JSON file:

```
$ cat demo_rspincs_train_output/training_history.json
```

The trained Keras model with the best validation loss is again saved in
`checkpoint/best-cp.ckpt` in the output dir.

# Notes

NOTE: the content of this research code repository
(i) is not intended to be a medical device; and
(ii) is not intended for clinical use of any kind, including but not limited to
diagnosis or prognosis.

This is not an officially supported Google product.
