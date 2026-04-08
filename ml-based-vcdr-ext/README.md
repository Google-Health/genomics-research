# Machine learning-based phenotyping for genomic discovery

## Overview

This repository contains a git patch extending the existing
[ML-based VCDR pipeline](https://github.com/Google-Health/genomics-research/blob/main/ml-based-vcdr).

## Installation

This extension exists as a git patch:

```text
genomics-research/ml-based-vcdr-ext/0001-Generate-a-patch-for-the-ML-based-VCDR-Extension-man.patch
```

In order to apply this patch to the existing ML-based VCDR pipeline, run:

```bash
$ git clone https://github.com/Google-Health/genomics-research.git
$ cd genomics-research
$ git am ml-based-vcdr-ext/0001-Generate-a-patch-for-the-ML-based-VCDR-Extension-man.patch
Applying: Generate a patch for the ML-based VCDR Extension manuscript.
```

After the patch has been successfully applied, you can then run the updated
ML-based VCDR pipeline following the existing instructions from that
[README](https://github.com/Google-Health/genomics-research/blob/main/ml-based-vcdr).

## Notes

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind,
including but not limited to diagnosis or prognosis.

This is not an officially supported Google product.
