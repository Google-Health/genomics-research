# DeepNull: Modeling non-linear covariate effects improves phenotype prediction and association power

This repository contains code implementing nonlinear covariate modeling to
increase power in genome-wide association studies, as described in "DeepNull:
Modeling non-linear covariate effects improves phenotype prediction and
association power"
([Hormozdiari et al 2021](https://doi.org/10.1101/2021.05.26.445783)).
The code is written using Python 3.7 and TensorFlow 2.4.

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

Of particular note is the `--model_config` flag. DeepNull uses the
[ml_collections](https://github.com/google/ml_collections) library to specify
all parameters related to the model and training regimen. The supported
configuration code is located in [`config.py`](config.py), and parameters can
be modified as described in detail in the
[`ml_collections README`](https://github.com/google/ml_collections#parameterising-the-get_config-function).
As a brief example, to use the DeepNull architecture with the `elu` activation
and train with batch size 4096, the above example command would be modified as
follows:

```bash
python -m deepnull.main \
  --input_tsv=/input/ORIGINAL_PHENOCOVAR_TSV \
  --output_tsv=/output/PHENOCOVAR_WITH_DEEPNULL_PREDICTION_TSV \
  --target=pheno \
  --covariates="age,sex,genotyping_array" \
  --model_config=/path/to/config.py:deepnull \
  --model_config.model_config.mlp_activation=elu \
  --model_config.training_config.batch_size=4096
```

where `/path/to/config.py` provides the path to [`config.py`](config.py) on your
machine.

## Incorporating DeepNull into a GWAS analysis

The above section, "How to run DeepNull", shows that the DeepNull software adds
a single column to a phenotype+covariate file of interest that represents a
nonlinear prediction of the target phenotype of interest. To incorporate this
into a GWAS analysis, the single additional covariate should be **added** as an
additional covariate. A concrete example with `BOLT-LMM`, using the same file,
phenotype `pheno`, and covariates `age`, `sex`, `genotyping_array` as above, is
shown below:

### Original example GWAS command
```bash
# N.B. Data loading flags are omitted for brevity.

bolt \
  --phenoFile /input/ORIGINAL_PHENOCOVAR_TSV \
  --covarFile /input/ORIGINAL_PHENOCOVAR_TSV \
  --qCovarCol age \
  --qCovarCol sex \
  --qCovarCol genotyping_array \
  --phenoCol pheno
```

After running DeepNull on the `/input/ORIGINAL_PHENOCOVAR_TSV` to create the new
TSV `/output/PHENOCOVAR_WITH_DEEPNULL_PREDICTION_TSV` that includes the column
`pheno_deepnull`, the updated command is given below:

### Updated GWAS command to incorporate DeepNull
```bash
# N.B. Data loading flags are omitted for brevity.
# Note the addition of the single `--qCovarCol pheno_deepnull` line.

bolt \
  --phenoFile /output/PHENOCOVAR_WITH_DEEPNULL_PREDICTION_TSV \
  --covarFile /output/PHENOCOVAR_WITH_DEEPNULL_PREDICTION_TSV \
  --qCovarCol age \
  --qCovarCol sex \
  --qCovarCol genotyping_array \
  --qCovarCol pheno_deepnull \
  --phenoCol pheno
```

## Data

Datasets used to reproduce the results from the above publication are available
to researchers with approved access to the
[UK Biobank](https://www.ukbiobank.ac.uk/).

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis or prognosis.

This is not an officially supported Google product.
