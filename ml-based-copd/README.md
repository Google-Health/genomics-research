# Leveraging deep-learning on raw spirograms to improve genetic understanding and risk scoring of COPD despite noisy labels

This repository contains code for training MLP and ResNet18 models to predict
COPD status from spirograms. The resulting ML-based COPD liability scores are
used in genome-wide association studies, as described in “Leveraging
deep-learning on raw spirograms to improve genetic understanding and risk
scoring of COPD despite noisy labels”
([Cosentino, Behsaz, Alipanahi, McCaw, *et al*., 2022](https://www.medrxiv.org/content/10.1101/2022.09.12.22279863v1)).
The code is written using Python 3.9 and TensorFlow 2.9.

Experiments are parameterized using the `ml_collections.ConfigDict`
configuration files in the `./learning/configs` subdirectory. This code example
illustrates the core implementation of the model architectures described in the
above publication, but omits the code needed to prepare and preprocess input
data as that is done on internal Google utilities.

Datasets used to reproduce the results from the above publication are available
to researchers with approved access to the
[UK Biobank](https://www.ukbiobank.ac.uk/). The ResNet18 ensemble member
TensorFlow checkpoints are available
[here](https://drive.google.com/drive/folders/1XZC9ByHBChDcQtNvS7RM60GqlOPTkocs).
GWAS summary statistics for the D-INT ML-based COPD liability score are
available
[here](https://drive.google.com/file/d/1Kpa64LmSjy_BUbF-a5vzq15OlMVkvrjh/view?usp=share_link).

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis or prognosis.

This is not an official Google product.

The code is not intended for active application to spirograms, but rather to
enable reproducibility of the analyses in the accompanying publication. It is
not intended to be a medical device and is not intended for clinical use of any
kind, including but not limited to diagnosis or prognosis.
