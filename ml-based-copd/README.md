# Inference of chronic obstructive pulmonary disease with deep learning on raw spirograms identifies novel genetic loci and improves risk models

## Overview

This repository contains code developed as part of the following paper:
“Inference of chronic obstructive pulmonary disease with deep learning
on raw spirograms identifies new genetic loci and improves risk models”
([Cosentino, Behsaz, Alipanahi, McCaw *et al*., *Nature Genetics*, 2023](https://www.nature.com/articles/s41588-023-01372-4)).

There are three pieces of functionality present in this repository:

1.  ResNet18 model training: code in `learning`
1.  ResNet18 model inference: code in `learning`
1.  Data analysis and figure generation: code in `analysis`

Both the training and inference code are executable given appropriate input data
(e.g. spirogram curves from UK Biobank). The analysis code contains subsets of
analyses that depend on specifically formatted data to be externally runnable,
but all the code has been provided for review and clear APIs for the all
functionality are provided. Details on expected schemas are available within
each analysis.

Model checkpoints, full ML-based COPD GWAS summary statistics, and annotated
hits, loci, and filtered results for internal GWAS are attached to the
[latest release](https://github.com/Google-Health/genomics-research/releases/tag/v0.2.0-ML-COPD).

Important: The code is not intended for active application to spirograms, but
rather to enable reproducibility of the analyses in the accompanying
publication. It is not intended to be a medical device and is not intended for
clinical use of any kind, including but not limited to diagnosis or prognosis.

This is not an official Google product.

## Installation

Installation supports Python 3.9 and TensorFlow 2.9. Follow the instructions at
https://www.tensorflow.org/install/gpu to set up GPU support for faster model
training and inference. Once GPU support is set up, install with
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) by executing
these instructions from the root of the checked-out repository:

```
conda create -n copd python=3.9
conda activate copd
pip install -r learning/requirements.txt
```

Note that the dependencies for
[colab notebooks](https://colab.research.google.com/) and
[R scripts](https://www.r-project.org/) in `analysis` are not included in
`learning/requirements.txt` and will need to be installed separately.

## Data sources

The datasets required to reproduce the results from the above publication are
available to researchers with approved access to the
[UK Biobank](https://www.ukbiobank.ac.uk/). The ResNet18 ensemble member
TensorFlow checkpoints are available
[here](https://github.com/Google-Health/genomics-research/releases/download/v0.2.0-ML-COPD/ml_based_copd_member_ckpts.zip).
GWAS summary statistics for the D-INT ML-based COPD liability score are
available
[here](https://github.com/Google-Health/genomics-research/releases/download/v0.2.0-ML-COPD/ml_based_copd.gwas.tsv.gz).
Annotated hits, loci, and filtered results for internal GWAS are also attached
to the
[latest release](https://github.com/Google-Health/genomics-research/releases/tag/v0.2.0-ML-COPD/).

## Dataset preprocessing

### Spirograms

See
[`ukb_3066_demo_preprocessing.py`](https://github.com/Google-Health/genomics-research/blob/main/ml-based-copd/learning/ukb_3066_demo_preprocessing.py)
for details on how raw UKB spirograms can be converted into flow-volume
representations compatible with the ML-based COPD model. This library converts
the single UKB demo spirometry blow showcased in field
[3066](https://biobank.ctsu.ox.ac.uk/crystal/field.cgi?id=3066). This spirometry
exhalation volume curve example is publicly available and can be downloaded
following the instructions on this page:
https://biobank.ndph.ox.ac.uk/ukb/refer.cgi?id=3. Accessing the full dataset
requires application and approval from the UK Biobank.

### COPD status

Binary COPD label definitions should be defined according to the label
definitions in Supplementary Table 1
([web link](https://www.nature.com/articles/s41588-023-01372-4#Sec25)). These
definitions are reproduced here for reference:

| Label                           | Definition                                                                                                                                                                                                                                     | Usage                                                                                                                                                                           |
|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Self report                     | Code 6 in field 6152 (medical conditions in touch screen questionnaires) or codes 1112, 1113, or 1472 in field 20002 (medical conditions in verbal interview).                                                                                 | Definition of medical-record-based COPD labels below used in training.                                                                                                          |
| Primary Care                    | ICD-10 codes J41, J42, J43, or J44 in field 42040 (GP clinical event records), after Read v2 and v3 codes in the records mapped into ICD-10 codes.                                                                                             | Definition of medical-record-based COPD labels below used in training.                                                                                                          |
| Training Hospitalization        | Includes primary or secondary causes of hospitalization. ICD-9 codes 491, 492, or 496 in field 41271 (Diagnoses - ICD9), or ICD-10 codes J41, J42, J43, or J44 in field 41270 (Diagnoses - ICD10).                                             | Definition of medical-record-based COPD labels below used in training.                                                                                                          |
| Medical-record-based            | If a COPD case in at least one of "self report", "primary care", and "training hospitalization" COPD labels.                                                                                                                                   | Training of ML models.                                                                                                                                                          |
| Evaluation medical-record-based | Logical OR of "self report", "primary care", and "training hospitalization" labels only when all three sources exist for an individual.                                                                                                        | Evaluation of ML models and GWAS hits. Having all three sources increases the likelihood of a correct COPD label which is preferred for evaluation. |
| Future hospitalization          | Only includes cases with COPD as primary cause of hospitalization after the spirometry test date. ICD-10 codes J41, J42, J43, or J44 in field 41234 (records in HES inpatient diagnoses dataset) after converting ICD-9 codes to ICD-10 codes. | Evaluation of ML models.                                                                                                                            |
| Death                           | ICD-10 J41, J42, J43, or J44 codes in field 40001 (primary cause of death).                                                                                                                                                                    | Evaluation of ML models and GWAS hits.                                                                                                              |
| Hospitalization                 | Similar to "future hospitalization" but also includes cases before the spirometry test date.                                                                                                                                                   | Evaluation of GWAS hits.                                                                                                                            |
| Proxy-GOLD                      | Mirrors moderate or worse GOLD grading for a single blow without bronchodilation: `FEV1/FVC<0.7` and `FEV1%pred<80%`.                                                                                                                          | Evaluating the noisiness of training labels, "medical-record-based" COPD. Evaluation of binarized ML-based COPD liability.                          |

Assuming one has extracted self report, primary care, and hospitalization label
columns for all individuals into a `copd_binary_df` pandas dataframe as
`copd_sr_src`, `copd_gp_src`, and `copd_hesin_src`, respectively, we then define
the medical-record-based COPD and evaluation medical-record-based COPD as
follows:

```python
import numpy as np
import pandas as pd


def float_logical_or(df: pd.DataFrame, columns: list[str]) -> pd.Series:
  # Assumes the values are 0.0, 1.0, or NaN. If all columns are NaN,
  # the result is NaN; otherwise, it is the max of non-NaN values, which is
  # equivalent to the logical or for 0/1 values.
  return df[columns].max(axis=1)


# Define medical-record-based COPD as the OR of self report, primary care, and
# hospitalization labels.
copd_binary_df['copd_mrb'] = float_logical_or(
    copd_binary_df,
    [
        'copd_sr_src',
        'copd_gp_src',
        'copd_hesin_src',
    ],
)

# Define evaluation medical-record-based COPD as the OR of self report, primary
# care, and hospitalization labels over individuals with non-missing values for
# all three sources.
has_all_srcs_mask = (
    copd_binary_df[[
        'copd_sr_src',
        'copd_gp_src',
        'copd_hesin_src',
    ]]
    .notna()
    .all(axis=1)
)
copd_binary_df['copd_mrb_eval'] = np.where(
    has_all_srcs_mask,
    copd_binary_df['copd_mrb'],
    np.nan,
)
```

The final label prevalence across dataset splits should be similar to that from
Supplementary Table 36
([web link](https://www.nature.com/articles/s41588-023-01372-4#Sec25)):

Dataset  | Split      | $n$    | MRB    | Eval. MRB | GOLD   | Hospitalization | Death
-------- | ---------- | ------ | ------ | --------- | ------ | --------------- | -----
Modeling | Train      | 259746 | 0.0383 | 0.0473    | 0.0725 | 0.0075          | 0.0007
Modeling | Validation | 65281  | 0.0388 | 0.0479    | 0.0720 | 0.0072          | 0.0007
Fold 1   | Train      | 128739 | 0.0384 | 0.0486    | 0.0711 | 0.0076          | 0.0007
Fold 1   | Validation | 32310  | 0.0391 | 0.0477    | 0.0719 | 0.0073          | 0.0008
Fold 2   | Train      | 129691 | 0.0379 | 0.0456    | 0.0734 | 0.0072          | 0.0007
Fold 2   | Validation | 32637  | 0.0381 | 0.0480    | 0.0714 | 0.0069          | 0.0006
PRS      | Holdout    | 110739 | 0.0640 | 0.0748    | -      | 0.0053          | 0.0021

Below is a detailed breakdown down of sample size and prevalence across label
sources:

label          | split | #cases | #samples | prevalence | % samples w/label
:------------- | :---- | -----: | -------: | ---------: | ----------------:
copd_mrb       | ALL   | 13104  | 351152   | 0.0373     | 1.0000
copd_sr_src    | ALL   | 5638   | 351040   | 0.0161     | 0.9997
copd_gp_src    | ALL   | 2934   | 161154   | 0.0182     | 0.4589
copd_hesin_src | ALL   | 8512   | 289280   | 0.0294     | 0.8238
copd_mrb_eval  | ALL   | 6227   | 133987   | 0.0465     | 0.3816
copd_mrb       | TRAIN | 9936   | 259746   | 0.0383     | 1.0000
copd_sr_src    | TRAIN | 4285   | 259707   | 0.0165     | 0.9998
copd_gp_src    | TRAIN | 2261   | 120425   | 0.0188     | 0.4636
copd_hesin_src | TRAIN | 6477   | 214743   | 0.0302     | 0.8267
copd_mrb_eval  | TRAIN | 4749   | 100505   | 0.0473     | 0.3869
copd_mrb       | VAL   | 2533   | 65281    | 0.0388     | 1.0000
copd_sr_src    | VAL   | 1075   | 65276    | 0.0165     | 0.9999
copd_gp_src    | VAL   | 571    | 30373    | 0.0188     | 0.4653
copd_hesin_src | VAL   | 1657   | 53776    | 0.0308     | 0.8238
copd_mrb_eval  | VAL   | 1212   | 25281    | 0.0479     | 0.3873

Here, `label` is the target label column, `split` is the dataset split, `#case`
is the total number of cases, `#samples` is the total number of samples with a
non-missing label, `prevalence` is the prevalence of the label, and `% samples
w/label` is the fraction of individuals with a non-missing label (i.e., either a
definitive case/control label is present). Note that the `ALL` split contains
*all* individuals with valid blows regardless of ancestry while the `TRAIN` and
`VAL` splits contain only European samples.

The `copd_gp_src` label is defined using UKB `gp_clinical` reads and the TRUD
mappings. At a high level, this requires the following:

1.  Download the [Read2](https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=1834)
    (`coding1834`) and
    [Read3](https://biobank.ndph.ox.ac.uk/ukb/coding.cgi?id=1835) (`coding1835`)
    coding TRUD mappings.
2.  Parse these to create coding->meaning maps (where coding is the Read2 and
    Read3 codings and meaning is the ICD-10 code). For example, there are a
    number of Read3 codes ("H310.", "H3101", "H310z", etc.) that map to the COPD
    "J41" code. You want to do this for all ICD 10 codes ('J41', 'J42', 'J43',
    and 'J44').
3.  Load your
    [GP clinical events](https://biobank.ndph.ox.ac.uk/ukb/rectab.cgi?id=1060)
    data TSV.
4.  Iterate over each line in the GP clinical data TSV (which contains EID, data
    provider, date, Read2, Read3, etc. columns), checking whether each event
    corresponds to one of the COPD-related ICD 10 codes. Build up an `{eid:
    has_copd_gp_src}` mapping denoting whether each EID has a COPD-related
    event.
5.  Convert this mapping into your `copd_gp_src` column.

Individuals who have COPD events are considered cases, individuals who have only
non-COPD events are considered controls, and individuals with no events are
considered missing (i.e., `nan`).

## Model training and inference

Experiments are parameterized using the `ml_collections.ConfigDict`
configuration files in the `./learning/configs` subdirectory. This code example
illustrates the core implementation of the model architectures described in the
above publication, but omits the code needed to prepare and preprocess input
data as that is done on internal Google utilities.

## Data analyses and figure generation

Details for running the following analyses are available in
[`learning/analysis_replication.md`](https://github.com/Google-Health/genomics-research/blob/main/ml-based-copd/learning/analysis_replication.md):

-   GWAS with
    [BOLT-LMM](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html)
-   Covariate interaction with
    [DeepNull](https://github.com/Google-Health/genomics-research/tree/main/nonlinear-covariate-gwas)
-   Heritability and genetic correlation with
    [LDSC](https://github.com/bulik/ldsc)
-   Enrichment with [GARFIELD](https://www.ebi.ac.uk/birney-srv/GARFIELD/)
-   Enrichment with [GREAT](http://great.stanford.edu/public/html/)
-   Survival analyses
-   PRS analyses
-   Plotting all main and extended data figures
