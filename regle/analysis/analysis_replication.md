# Replicates all main analyses in the REGLE paper

## Analysis of the embeddings

1. See `embedding_interpretability.ipynb`.


## Principal component analysis (PCA) and spline fitting

See `pca_and_spline_fitting.ipynb`.


## GWAS

1. GWAS on all phenotypes via [BOLT-LMM](https://alkesgroup.broadinstitute.org/BOLT-LMM/BOLT-LMM_manual.html):

    ```[bash]
    PHENO_NAME="..."
    PHENO_FILE="..."
    BOLT_LDSC_DIR="..."
    UKB_GENOTYPED_DIR="..."
    UKB_IMPUTED_DIR="..."
    UKB_BGEN_DIR="..."
    bolt \
      --numThreads 64 \
      --LDscoresFile "${BOLT_LDSC_DIR}/LDSCORE.1000G_EUR.tab.gz" \
      --LDscoresMatchBp \
      --covarFile "${PHENO_FILE}" \
      --phenoFile "${PHENO_FILE}" \
      --phenoCol "${PHENO_NAME}" \
      --statsFile /tmp/tmp_result_experiment1 \
      --fam "${UKB_GENOTYPED_DIR}/all_samples.fam" \
      --sampleFile "${UKB_IMPUTED_DIR}/ukb.sample" \
      --predBetasFile /tmp/genotyped_variants.betas \
      --remove "${UKB_GENOTYPED_DIR}/nonoverlapping_samples.txt" \
      --lmmForceNonInf \
      --bgenMinMAF 9.999999747378752e-05 \
      --bgenMinINFO 0.6000000238418579 \
      --bgenFile "${UKB_BGEN_DIR}/ukb_imp_chr10_v3_mininfo_0.6.bgen" \
      --statsFileBgenSnps /tmp/tmp_bgen_result_experiment1 \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr10_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr11_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr12_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr13_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr14_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr15_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr16_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr17_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr18_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr19_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr1_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr20_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr21_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr22_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr2_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr3_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr4_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr5_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr6_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr7_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr8_v2.bed" \
      --bed "${UKB_GENOTYPED_DIR}/ukb_cal_chr9_v2.bed" \
      --qCovarCol age \
      --qCovarCol age_x_age \
      --qCovarCol age_x_sex \
      --qCovarCol bmi \
      --qCovarCol genotyping_array \
      --qCovarCol height_cm \
      --qCovarCol height_cm_x_height_cm \
      --qCovarCol model_fold \
      --qCovarCol occasional_smoker \
      --qCovarCol pc1 \
      --qCovarCol pc10 \
      --qCovarCol pc11 \
      --qCovarCol pc12 \
      --qCovarCol pc13 \
      --qCovarCol pc14 \
      --qCovarCol pc15 \
      --qCovarCol pc2 \
      --qCovarCol pc3 \
      --qCovarCol pc4 \
      --qCovarCol pc5 \
      --qCovarCol pc6 \
      --qCovarCol pc7 \
      --qCovarCol pc8 \
      --qCovarCol pc9 \
      --qCovarCol sex \
      --qCovarCol smoker \
      --qCovarCol smoking_pack_per_year \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr10_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr11_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr12_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr13_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr14_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr15_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr16_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr17_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr18_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr19_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr1_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr20_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr21_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr22_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr2_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr3_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr4_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr5_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr6_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr7_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr8_v2.bim" \
      --bim "${UKB_GENOTYPED_DIR}/ukb_cal_chr9_v2.bim"
    ```


## [LDSC](https://github.com/bulik/ldsc)

1. Run munge:

    ```[bash]
    BOLT_GWAS_FILE="..."
    LDSC_INPUT_DIR="..."
    LDSC_OUTPUT_DIR="..."
    source activate ldsc && python /opt/ldsc/munge_sumstats.py \
      --sumstats "${BOLT_GWAS_FILE}" \
      --merge-alleles "${LDSC_INPUT_DIR}/w_hm3.snplist" \
      --out "%{LDSC_OUTPUT_DIR}/munge \
      --chunksize 500000
    ```

1. Run S-LDSC:

    ```[bash]
    source activate ldsc && python /opt/ldsc/ldsc.py \
      --h2 "${LDSC_OUTPUT_DIR}/munge.sumstats.gz" \
      --ref-ld-chr "${LDSC_INPUT_DIR}/baselineLD." \
      --w-ld-chr "${LDSC_INPUT_DIR}/weight." \
      --out "${LDSC_OUTPUT_DIR}/ldsc"
    ```

## [GARFIELD](https://www.ebi.ac.uk/birney-srv/GARFIELD/)

1. For each chromosome run:
    ```[bash]
    GARFIELD_INPUT_DIR="..."
    GARFIELD_OUTPUT_DIR="..."
    ANNOTATION_LIKE_FILE="..."
    INPUT_FILE_P="..."
    ./garfield/garfield-prep-chr \
      -ptags "${GARFIELD_INPUT_DIR}/tags/r01/*"\
      -ctags "${GARFIELD_INPUT_DIR}/tags/r08/*" \
      -maftss "${GARFIELD_INPUT_DIR}/maftssd/*"\
      -pval "${INPUT_FILE_P}"\
      -ann "${GARFIELD_INPUT_DIR}/annotation/*"\
      -excl -1\
      -chr "${CHR}" \
      -o "${GARFIELD_OUTPUT_DIR}/tmp_prep_out"
    ```

1. For each chromosome run:
    ```[bash]
    Rscript garfield-Meff-Padj.R \
     -i "${GARFIELD_OUTPUT_DIR}/tmp_prep_out"\
     -o "${GARFIELD_OUTPUT_DIR}/tmp_meff_out"
    ```

1. To compute enrichment:
    ```[bash]
    Rscript garfield-test.R \
      -i "${GARFIELD_OUTPUT_DIR}/tmp_prep_out" \
      -o "${GARFIELD_OUTPUT_DIR}/tmp_test_out" \
      -l "${ANNOTATION_LIKE_FILE}" \
      -pt 1e-5,1e-8\
      -b m5,n5,t5\
      -s 1-1005 \
      -c 0
    ```

1. Plotting
    ```[bash]
    Rscript garfield-plot.R \
      -i "${GARFIELD_OUTPUT_DIR}/tmp_prep_out" \
      -o "${GARFIELD_OUTPUT_DIR}/tmp_plot_out" \
      -l "${ANNOTATION_LIKE_FILE}" \
      -t " "\
      -f 10 \
      -padj "${PVAL_ADJ}"
    ```


## Polygenic risk score (PRS) analysis

Given the effect sizes computed by BOLT-LMM or by "pruning and thresholding" as
described in the paper, we generated each individual's polygenic risk scores
(PRS) using [PLINK](https://www.cog-genomics.org/plink/2.0/) as follows.

1. We ran PLINK to compute the PRS by the following command:
    ```[bash]
    plink \
    --bed $BED_FILE \
    --bim $BIM_FILE \
    --fam $FAM_FILE \
    --read-freq $VARIANT_FREQ_FILE \
    --score ${MODEL_FILE} header sum double-dosage \
    --out $PLINK_OUT
    ```

1. See `prs_analysis.ipynb` to compute various PRS metrics we use in the paper
using (paired) bootstrapping.
