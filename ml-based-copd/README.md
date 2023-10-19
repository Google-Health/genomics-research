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
