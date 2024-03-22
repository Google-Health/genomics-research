# M-REGLE (Multimodal REpresentation learning for Genetic discovery on Low-dimensional Embeddings)

This repository contains code for defining and training models for joint
representations of cardiovascular data modalities (12-lead ECG and ECG lead I +
PPG), using the M-REGLE ("Multimodal REpresentation learning for Genetic
discovery on Low-dimensional Embeddings") framework described in this
manuscript:
"[Utilizing multimodal AI to improve genetic analyses of cardiovascular traits](https://www.medrxiv.org/content/10.1101/2024.03.19.24304547v1)".
It also includes the trained model checkpoints used for the experiments in the
manuscript.

Please cite the above manuscript if you use this repository for your research.

```
@article {Zhou2024.03.19.24304547,
	author = {Yuchen Zhou and Justin T Cosentino and Taedong Yun and Mahantesh I Biradar and Jacqueline Shreibati and Dongbing Lai and Tae-Hwi Schwantes-An and Robert Luben and Zachary R McCaw and Jorgen Engmann and Rui Providencia and Amand Floriaan Schmidt and Patricia B. Munroe and Howard Yang and Andrew Carroll and Anthony P Khawaja and Cory McLean and Babak Behsaz and Farhad Hormozdiari},
	title = {Utilizing multimodal AI to improve genetic analyses of cardiovascular traits},
	elocation-id = {2024.03.19.24304547},
	year = {2024},
	doi = {10.1101/2024.03.19.24304547},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/03/21/2024.03.19.24304547},
	eprint = {https://www.medrxiv.org/content/early/2024/03/21/2024.03.19.24304547.full.pdf},
	journal = {medRxiv}
}
```


## Running demo using the trained model

The following steps will generate a demo dataset consisting of a single ECG Lead
I + PPG sample, and calculate a joint representation of that using the trained
VAE model. We use publicly available examples that can be downloaded from the UK
Biobank webpage: https://biobank.ndph.ox.ac.uk/showcase/refer.cgi?id=20205,

First, install required packages. (We recommend using a virtual environment.)

```
$ pip3 install -r requirements.txt
```

Second, generate a demo dataset in `data` directory.

```
$ python3 generate_dataset_from_ukb_demo.py \
  --out_dir=data/ \
  --dataset=ecgppg
```

Third, generate joint representations (dim=12) using the demo dataset:

```
$ python3 generate_mregle_embeddings.py \
  --output_dir=/path/to/output \
  --dataset=ecgppg
```

Similarly, create a dataset of a 12-lead ECG example and generate joint
representation of it using the demo dataset and print output:

```
$ python3 generate_dataset_from_ukb_demo.py --out_dir=data/ --dataset=ecg12

$ python3 generate_mregle_embeddings.py \
  --output_dir=/path/to/output \
  --dataset=ecg12
```

## Training a M-REGLE model using your data

For the demo training, we first generate fake train and validation data by
duplicating the demo data multiple times. In practice, you can use your own
12-lead ECGs or LEAD I ECG + PPG waveforms to train the model by following the
similar steps.

```
$ mkdir demo_train
$ mkdir demo_val
$ python3 generate_dataset_from_ukb_demo.py \
  --out_dir=demo_train/ \
  --dataset=ecgppg \
  --duplicates=80

$ python3 generate_dataset_from_ukb_demo.py \
  --out_dir=demo_val/ \
  --dataset=ecgppg \
  --duplicates=20
```

Now we are ready to train M-REGLE models using the demo datasets.

### Training a M-REGLE model on lead I ECG + PPG

We use `train.py` to train a VAE model for the given ECG and PPG waveforms. The
argument `latent_dim` determines the dimension of the latent variables. We used
12 dimensions in the paper, but you are free to choose other values. Here we use
the default settings for the model structure. You can additionally pass your
hyperparameters such as `random_seed`, `learning_rate`, `batch_size`, and
`num_epochs`.

```
$ python3 train.py \
  --logging_dir=log \
  --data_setting=ecgppg \
  --train_data_path=demo_train/ecgppg_ml_data.npy \
  --validation_data_path=demo_val/ecgppg_ml_data.npy \
  --latent_dim=12
```

### Training a M-REGLE model on 12-lead ECG

Similarly, first generate demo datasets.
```
$ mkdir demo_train
$ mkdir demo_val
$ python3 generate_dataset_from_ukb_demo.py \
  --out_dir=demo_train/ \
  --dataset=ecg12 \
  --duplicates=80

$ python3 generate_dataset_from_ukb_demo.py \
  --out_dir=demo_val/ \
  --dataset=ecg12 \
  --duplicates=20
```

Then run `train.py` script to start training.

```
$ python3 train.py \
  --logging_dir=log \
  --data_setting=ecg12 \
  --train_data_path=demo_train/ecg_ml_data.npy \
  --validation_data_path=demo_val/ecg_ml_data.npy \
  --latent_dim=96
```

## Notes

NOTE: the content of this research code repository
(i) is not intended to be a medical device; and
(ii) is not intended for clinical use of any kind, including but not limited to
diagnosis or prognosis.

This is not an officially supported Google product.