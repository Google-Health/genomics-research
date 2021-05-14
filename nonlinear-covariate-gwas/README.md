# DeepNull: Leveraging non-linear covariate effects to improve power in genetic association tests

This repository contains code implementing nonlinear covariate modeling to
increase power in genome-wide association studies, as described in "DeepNull:
Modeling non-linear covariate effects improves phenotype prediction and
association power" (Hormozdiari et al 2021). The code is written using Python
3.7 and TensorFlow 2.4.

## Installation

Installation is not required to run DeepNull end-to-end; you can just
[open `DeepNull_e2e.ipynb` in colab](https://colab.research.google.com/github/Google-Health/genomics-research/blob/main/nonlinear-covariate-gwas/DeepNull_e2e.ipynb)
to try it out.

To install DeepNull locally, run

```bash
pip install --upgrade pip
pip install --upgrade deepnull
```

on a machine with Python 3.7+. This installs a CPU-only version, as there are
typically few enough covariates that using accelerators does not provide
meaningful speedups.

## How to run DeepNull

To run locally, there is a single required input file. This file contains the
phenotype of interest and covariates used to predict the phenotype, formatted as
a *tab-separated* file suitable for GWAS analysis with
[PLINK](https://www.cog-genomics.org/plink/2.0/assoc) or
[BOLT-LMM](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html).

Briefly, the file must contain a single header line. The first two columns must
be `FID` and `IID`, and all `IID` values must be unique.

An example command to train DeepNull to predict the phenotype `pheno` from
covariates `age`, `sex`, and `genotyping_array` is the following:

```bash
python -m deepnull.main \
  --input_tsv=/input/YOUR_PHENOCOVAR_TSV \
  --output_tsv=/output/YOUR_OUTPUT_TSV \
  --target=pheno \
  --covariates="age,sex,genotyping_array"
```

To see all available flags, run

```bash
python -m deepnull.main --help 2> /dev/null
```

## Data

Datasets used to reproduce the results from the above publication are available
to researchers with approved access to the
[UK Biobank](https://www.ukbiobank.ac.uk/).

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis or prognosis.

This is not an officially supported Google product.
