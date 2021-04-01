# Machine learning-based phenotyping for genomic discovery

## Overview

This repository contains code developed as part of the following paper:

* ["Large-scale machine learning-based phenotyping significantly improves
genomic discovery for optic nerve head morphology"](https://arxiv.org/abs/2011.13012) (Alipanahi *et al*., 2021).

There are three pieces of functionality present in this repository:

1.  Model training: code in `learning`
2.  Model inference: code in `learning`
3.  Data analysis and figure generation: code in `analysis`

The sections below describe these functionalities in detail. Both the training
and inference code are executable given appropriate input data (e.g. fundus
images from UK Biobank). The data analysis code contains subsets of analyses
that depend on code from the main Google code repository that make it not
externally runnable, but all the code has been provided for review and clear
APIs for the missing functionality are provided.

Model predictions for vertical cup-to-disc ratio, glaucoma referral risk, image
gradability, and vertical cup-to-disc visibility for all UK Biobank fundus
images, as well as the entire set of summary statistics from the GWAS analyses
performed in the manuscript, will be deposited into UK Biobank for open access
to all UK Biobank investigators upon publication.

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis or prognosis.

This is not an officially supported Google product.

## Installation

Installation supports Python 3.7+. Follow the instructions at
https://www.tensorflow.org/install/gpu to set up GPU support for faster model
training. Once GPU is set up, install with
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) by executing
these instructions from the root of the checked-out repository:

```
conda create -n vcdr python=3.7
conda activate vcdr
pip install -r learning/requirements.txt
```

## Model training

### Datasets

To train the model, the data (paired fundus images and their labels) must be in
[`TFRecord`](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecords_format_details)
format. Fundus images should be cropped and centered and have aspect ratio of
one. (The training pipeline can resize the images to `587x587`, but we recommend
resizing the images before generating the `TFRecord` datasets to improve
training speed.) The schema of a single record is as follows:

```python
{
    'image/encoded':
        FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image/id':
        FixedLenFeature(shape=[1], dtype=tf.string, default_value=None),
    'image/glaucoma_gradability/value':
        FixedLenFeature(shape=[3], dtype=tf.float32, default_value=None),
    'image/glaucoma_gradability/weight':
        FixedLenFeature(shape=[1], dtype=tf.float32, default_value=None),
    'image/glaucoma_suspect_risk/value':
        FixedLenFeature(shape=[4], dtype=tf.float32, default_value=None),
    'image/glaucoma_suspect_risk/weight':
        FixedLenFeature(shape=[1], dtype=tf.float32, default_value=None),
    'image/vertical_cd_visibility/value':
        FixedLenFeature(shape=[3], dtype=tf.float32, default_value=None),
    'image/vertical_cd_visibility/weight':
        FixedLenFeature(shape=[1], dtype=tf.float32, default_value=None),
    'image/vertical_cup_to_disc/value':
        FixedLenFeature(shape=[1], dtype=tf.float32, default_value=None),
    'image/vertical_cup_to_disc/weight':
        FixedLenFeature(shape=[1], dtype=tf.float32, default_value=None)
}
```

The `image/{outcome}/weight` fields denote whether the image has the label for
`{outcome}` or if it is missing. For example, if an image is labeled as
ungradable, it will not have a `glaucoma_suspect_risk` label and its
`image/glaucoma_suspect_risk/weight` will be 0.

`vertical_cup_to_disc` is a number between 0 and 1, while other outcomes are
categorical with the following categories (for a detailed decription of these
categories see
[Phene _et al._ 2019](https://www.aaojournal.org/article/S0161-6420\(19\)31875-5/fulltext)):

```python
{
    'glaucoma_gradability': {
        0: 'UNGRADABLE',
        1: 'WITH_DIFFICULTY',
        2: 'GRADABLE',
    },
    'glaucoma_suspect_risk': {
        0: 'NON_GLAUCOMATOUS',
        1: 'LOW_RISK',
        2: 'HIGH_RISK',
        3: 'LIKELY',
    },
    'vertical_cd_visibility': {
        0: 'UNABLE_TO_ASSESS',
        1: 'COMPROMISED',
        2: 'SUFFICIENT',
    },
}
```

Note: The pipeline assumes that train, validation, test, and prediction
`TFRecord` files are located at the following paths:

```python
config.dataset = ml_collections.ConfigDict({
    'train': '/mnt/disks/data/train/train.tfrecord*',
    'eval': '/mnt/disks/data/train/eval.tfrecord*',
    'test': '/mnt/disks/data/train/test.tfrecord*',
    'predict': '/mnt/disks/data/predict/predict.tfrecord*',
    ...
})
```

Prior to running any portion of the pipeline, update these paths in the
experimental configuration, e.g.,
[`learning/configs/base.py`](learning/configs/base.py).

### Training

The training pipeline loads the data, resizes the images if needed, initilizes
the Inception V3 model with ImageNet parameters, applies image augmentation and
then learns the model parameters.

To train a single model, run the following command:

```bash
python3 train.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr
```

In total, 10 models are independently trained and the initialization of the top
layer of the model and the ordering of the training examples are different for
each model. During the training, we evaluate on the `validation` dataset every
500 steps and checkpoint if it is the best model based on the mean squared error
(MSE) of `vertical_cup_to_disc` we have seen thus far. After the training steps
are exhausted, we export the best checkpoint. The ultimate output of the model
is the average of outputs of the 10 models.

To train a full ensemble, run the following command:

```bash
python3 train_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base \
    --num_members=10 \
    --main_seed=42
```

Note: [`train_ensemble.py`](learning/train_ensemble.py) assumes that the
`num_members` models are trained sequentially on a single GPU. Assuming that `N`
GPUs are available, `N` models can be trained in parallel. See
`train_ensemble.py` for details on how to shard ensemble training across
multiple GPUs and recombine member checkpoints for ensemble validation and
inference.

## Model evaluation and inference

Both individual members and full ensembles can be evaluated on the train,
validation, and test dataset splits. To evaluate a single model or an individual
member, run one of the following commands:

```bash
# Evaluating a single model trained using `train.py`
python3 evaluate.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr \
    --split=EVAL

# Evaluating an ensemble member trained using `train_ensemble.py`
python3 evaluate.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base/member_9 \
    --split=EVAL
```

Similarly, the [`evaluate_ensemble.py`](learning/evaluate_ensemble.py) script
can be used to evaluate an ensemble of member models:

```bash
python3 evaluate_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base \
    --evaluate_members=True
```

Note: When evaluating an ensemble, the provided workdir should correspond to the
ensemble's base directory, rather than a member's subdirectory (i.e.,
`base_workdir` rather than `base_workdir/member_{i}`).

Use the [`predict.py`](learning/predict.py) and the
[`predict_ensemble.py`](learning/predict_ensemble.py) scripts to perform model
inference on new images. The `predict` dataset should be stored using `TFRecord`
files, similarly to those images used for training. However, these `TFRecord`s
only need to include the image and image id keys:

```python
{
    'image/encoded':
        FixedLenFeature(shape=[], dtype=tf.string, default_value=None),
    'image/id':
        FixedLenFeature(shape=[1], dtype=tf.string, default_value=None),
}
```

Once the input images have been properly formatted, run inference with one of
the following commands:

```
# Generating predictions with a single model trained using `train.py`
python3 predict.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr \
    --output_filepath=./outcome_predictions.csv

# Generating predictions with an ensemble member trained using
# `train_ensemble.py`
python3 predict.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base/member_9 \
    --output_filepath=./outcome_predictions.csv

# Generating predictions using all ensemble members
python3 predict_ensemble.py \
    --config=configs/base.py \
    --workdir=./reproduce_vcdr_ensemble_base \
    --output_filepath=./outcome_predictions.csv
```

These scripts save model predictions as a CSV, using a unique
`predict_utils.ID_KEY` ID column corresponding to each input image. The CSV
contains a column for each classification and regression target, i.e., a
multiclass outcome of shape `(None, N)` will have `N` columns. See
`predict_utils.OUTCOME_COLUMN_MAP` for column label mappings. Assuming that the
model is trained using the heads specified in
[`configs/base.py`](learning/configs/base.py), the `--output_filepath` CSV will
be formatted as follows:

```text
$ head -n ./outcome_predictions.csv
image_id,glaucoma_gradability:UNGRADABLE,glaucoma_gradability:WITH_DIFFICULTY,glaucoma_gradability:GRADABLE,glaucoma_suspect_risk:NON_GLAUCOMATOUS,glaucoma_suspect_risk:LOW_RISK,glaucoma_suspect_risk:HIGH_RISK,glaucoma_suspect_risk:LIKELY,vertical_cd_visibility:UNABLE_TO_ASSESS,vertical_cd_visibility:COMPROMISED,vertical_cd_visibility:SUFFICIENT,vertical_cup_to_disc:VERTICAL_CUP_TO_DISC
'unique_image_id_0',0.004463,0.8257,0.17,0.8584,0.1106,0.02357,0.007484,0.808,0.1862,0.005688,0.3887
```

### Performing inference with the model used in Alipanahi et al.

Individuals interested in performing inference on fundus images with the model
used to generate results in Alipanahi *et al.* for the purposes of scientific
reproducibility or generalizability of the work should email cym@google.com
for details.

## Data analysis and figure generation

The code and analyses used to create figures in the above manuscript are
provided in `analysis` and named accordingly. Expected input data formats and
locations are specified at the top of each module. There are three Jupyter
Notebooks in `analysis`, which we review in detail below.

### Phenotype calling

After the inference stage, we have predictions for 175,337 fundus images, from
the maximum of two possible imaging visits, and we call the phenotypes as
implemented in
[analysis/phenotype\_calling.ipynb](analysis/phenotype_calling.ipynb).

#### VCDR

We define the phenotype only based on one of these visits. We define the VCDR
phenotype as follows:

1.  We define gradability for the VCDR phenotype as both `glaucoma_gradability`
    and `vertical_cd_visibility` larger than 0.7 and remove any fundus image
    that does not satisfy both of these requirements.

1.  Since there is a ~5 year difference between the two visits, and VCDR can
    change over time due to many factors including age, medications, and eye
    operations, we prefer to use measurements from the first visit over the
    second one where possible. Therefore, if an individual has any gradable
    image(s) from visit 1, we define the phenotype based on these images;
    otherwise, we define it based on visit 2 (a.k.a. first repeat imaging
    visit). The final phenotype is called `vcdr_visit`.

1.  For a specific visit, we first average the VCDRs of each eye and then
    average these per eye VCDRs if both eyes have gradable images.

1.  To account for the impact of image gradability on the phenotype, we computed
    the average gradability score `gradability_visit` of all images used in
    defining an individual's phenotype. Moreover, we define `visit_age` as the
    age at the time of the visit used for computing `vcdr_visit`.

1.  To control for the small variations in phenotype calling, we add the visit
    number used `visit` (i.e., 1 or 2) and the number of eyes used `num_eyes` in
    calling the phenotype (i.e., 1 or 2) as covariates.

1.  Lastly, we subset to individuals of European ancestry.

#### Glaucoma liability

Since glaucoma could start from one of the eyes, we define the glaucoma
liability phenotype by taking the maximum risk across all fundus images of an
individual. We define this phenotype as follows:

1.  We define gradability as `glaucoma_gradability` larger than 0.7 and remove
    any fundus image that does not satisfy this requirement.

1.  Glaucoma referral risk has four levels and we pick the highest level
    `likely_glaucoma`, which is a probability between 0 and 1 as the raw risk.

1.  We compute the glaucoma referral risk as the maximum risk across all
    gradable fundus images for an individual.

1.  We then define glaucoma liability as the logit of the glaucoma risk.

1.  Lastly, we subset to individuals of European ancestry.

## Learning polygenic risk scores (PRS)

Using the ML-based VCDR and
[Craig _et al._ 2020](https://pubmed.ncbi.nlm.nih.gov/31959993/) GWAS hits, we
developed two sets of PRS in the
[analysis/learning\_prs.ipynb](analysis/learning_prs.ipynb) notebook as
described below.

### Test sets

We used two held out test sets for assessing the performance of different PRS:

1.  The UK Biobank evaluation set consisted of adjudicated expert-annotated VCDR
    measurements in 2,076 individuals of European ancestry.

2.  The EPIC-Norfolk evaluation set consisted of scanning laser ophthalmoscopy
    (HRT)-measured VCDRs in 5,868 individuals.

All predictions were made using [PLINK](https://www.cog-genomics.org/plink/) via
the `plink --score` command and performance metrics were computed using the
scores in the resulting `*.profile` files.

### Pruning and thresholding

Pruning and thresholding-based polygenic risk scores for VCDR were computed as
the weighted sum of effect allele counts for independent genome-wide significant
variants (`P≤5e-8`), where the weight of each variant was its estimated effect
size from the GWAS. To evaluate performance both within the UK Biobank and in
the EPIC-Norfolk cohorts, index variants present in both cohorts were used in
PRS creation, resulting in 58 of the 76 published variants from Craig *et al.*
GWAS and 282 of the 299 index variants from the ML-based GWAS.

### Elastic Net

Elastic net-based polygenic risk scores for VCDR were trained using the
ML-predicted VCDR as the target label in 62,969 individuals using
`scikit-learn`. The Craig *et al.* model used 76 variants (the 58 described in
the pruning and thresholding section above, plus 18 proxy variants present in
both UK Biobank and EPIC-Norfolk that were in highest linkage disequilibrium
(`R2≥0.6`) with the 18 dropped Craig *et al.* variants) and the ML-based model
used the same 282 variants as described above. Each model was trained with
5-fold cross-validation and L1-penalty ratios of `[0.1, 0.5, 0.7, 0.9, 0.95,
0.99, 1.0]`.

### Permutation *P* values

A permutation test was applied to assess whether a PRS trained using summary
statistics from the ML-based GWAS significantly outperformed a PRS trained using
summary statistics from the Craig *et al.* GWAS for predicting VCDR in the UK
Biobank and EPIC-Norfolk cohorts (`get_permute_pvalue`). Phenotypic predictions
were generated from both PRS. The test statistic was the difference in Pearson
correlations between the observed and predicted phenotypes, comparing ML-based
with Craig *et al.* A value exceeding zero indicates better performance by the
ML-based PRS. Under the null hypothesis, the predictions from both PRS are
exchangeable. To obtain a realization from the null distribution, for each
subject, the predictions of the ML-based and Craig *et al.* PRS were randomly
swapped, and the difference in correlations was recalculated. This procedure was
repeated `1e5` times to obtain the null distribution. The one-sided *P* value is
given by the proportion of realizations from the null distribution that were as
or more extreme than the observed difference in correlations.

## Plotting figures

The code used for plotting figures in the manuscript is in the
[analysis/plotting\_results.ipynb](analysis/plotting_results.ipynb) notebook.
