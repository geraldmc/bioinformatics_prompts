"""Module containing the GWAS discipline prompt template."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a GWAS prompt template
gwas_prompt = BioinformaticsPrompt(
    discipline="Genome-Wide Association Studies",
    description=(
        "Genome-Wide Association Studies (GWAS) identify statistical associations between genetic variants and traits or diseases "
        "across the genome. Modern GWAS approaches have evolved beyond simple association testing to incorporate functional genomics, "
        "machine learning, and polygenic risk modeling. These advanced methods help researchers address challenges such as multiple "
        "testing correction, population stratification, and the biological interpretation of identified variants. By integrating diverse "
        "data types and sophisticated analytical approaches, GWAS can provide insights into the genetic architecture of complex traits "
        "and contribute to precision medicine initiatives."
    ),
    key_concepts=[
        "Single nucleotide polymorphisms (SNPs) and genetic variation",
        "Statistical association testing and significance thresholds",
        "Multiple testing correction (Bonferroni, FDR, permutation-based methods)",
        "Population stratification and correction methods (PCA, mixed models)",
        "Linkage disequilibrium (LD) and haplotype analysis",
        "Polygenic risk scores (PRS) and genetic risk prediction",
        "Fine-mapping and causal variant identification",
        "Functional annotation and pathway analysis",
        "Gene-environment interactions (GxE)",
        "Meta-analysis and cross-population studies"
    ],
    common_tools=[
        "PLINK - GWAS analysis toolkit",
        "GCTA - Genome-wide Complex Trait Analysis",
        "BOLT-LMM - Linear mixed models for association testing",
        "SAIGE - Scalable and accurate implementation of generalized mixed models",
        "LDSC - LD Score Regression for heritability and genetic correlation",
        "METAL - Meta-analysis of genome-wide association scans",
        "PAINTOR/CAVIARBF/FINEMAP - Fine-mapping tools",
        "PRS-CS/LDpred - Polygenic risk score calculation",
        "ANNOVAR/VEP - Variant annotation",
        "FUMA - Functional mapping and annotation"
    ],
    common_file_formats=[
        {"name": "PLINK (.bed/.bim/.fam)", "description": "Binary format for genotype data"},
        {"name": "VCF", "description": "Variant Call Format for genetic variation data"},
        {"name": "BGEN/BGENv1.2", "description": "Binary format for imputed genetic data"},
        {"name": "GEN/SAMPLE", "description": "Oxford-format imputed genetic data"},
        {"name": "GWAS summary statistics", "description": "Text files with SNP association results"},
        {"name": "QC report formats", "description": "Quality control metrics and plots"},
        {"name": "Manhattan/QQ plots", "description": "Standard visualization formats in GWAS"}
    ],
    examples=[
        FewShotExample(
            query="What's the best approach for conducting a GWAS and addressing multiple testing correction?",
            context="The user has genotype data from 5,000 cases and 5,000 controls and wants to identify variants associated with a disease.",
            response="""
# Conducting a GWAS with Robust Multiple Testing Correction

## Approach Overview
A comprehensive GWAS with proper multiple testing correction involves:
1. Rigorous quality control of genetic data
2. Population structure assessment and correction
3. Association testing with appropriate statistical models
4. Multiple testing correction with modern approaches
5. Post-GWAS analysis and biological interpretation

## Key Decision Points

### Quality Control (QC) Strategy
- **Sample-level QC**:
  - Missing genotype rate (exclude samples with >2-5% missing data)
  - Sex discrepancy checks (compare genetic vs. reported sex)
  - Heterozygosity rate (exclude outliers ±3 SD from mean)
  - Relatedness (remove samples with PI_HAT >0.185, indicating 3rd-degree relatives)
  - Ancestry outliers (PCA-based identification)

- **Variant-level QC**:
  - Missing call rate (exclude SNPs with >2-5% missing data)
  - Minor allele frequency (typically exclude MAF <0.01 or 0.005)
  - Hardy-Weinberg equilibrium (HWE p < 1e-6 in controls)
  - Differential missingness between cases/controls (p < 1e-5)
  - Batch effects assessment if multiple genotyping batches

- **Special considerations**:
  - More stringent thresholds for rare variant analysis
  - Less stringent HWE for case-only variants (potential disease associations)
  - Chromosome-specific QC for sex chromosomes

### Population Structure Correction
- **Principal Component Analysis (PCA)**:
  - Use linkage-disequilibrium (LD) pruned variants (r² <0.2)
  - Exclude regions with long-range LD (MHC, chromosome inversions)
  - Typically include 5-20 PCs as covariates in association models
  - Consider global ancestry matching in diverse populations

- **Linear Mixed Models (LMM)**:
  - More powerful than PC correction in structured populations
  - Accounts for both population structure and cryptic relatedness
  - Methods like BOLT-LMM, SAIGE, or GCTA-MLMA
  - Computationally intensive but recommended for diverse populations

- **Alternative approaches**:
  - Genomic control (λGC) (older method, less recommended now)
  - Meta-analysis of ancestry-specific results
  - Distance-based approaches for extremely diverse cohorts

### Association Testing Strategy
- **Binary Trait Models**:
  - Logistic regression with PC covariates
  - Mixed logistic regression for related individuals
  - Firth logistic regression for rare variants/separation issues
  - SAIGE for case-control imbalance

- **Quantitative Trait Models**:
  - Linear regression with PC covariates
  - Linear mixed models for related individuals
  - Consider appropriate transformations for non-normal traits
  - Robust regression for outlier handling

- **Special Case Models**:
  - Ordinal regression for ordered categorical traits
  - Survival models for time-to-event outcomes
  - Count models (Poisson, negative binomial) for count data

### Multiple Testing Correction Approaches
- **Standard corrections**:
  - Bonferroni correction (p < 5e-8 for genome-wide significance)
  - False Discovery Rate (FDR, Benjamini-Hochberg)
  - Permutation-based family-wise error rate (FWER)

- **Modern refinements**:
  - LD-aware multiple testing correction
  - Stratified FDR based on functional annotations
  - Bayesian approaches (e.g., Bayes factors)

- **Candidate region approaches**:
  - Region-specific significance thresholds based on LD structure
  - Enhanced power through reduced testing burden
  - Useful for fine-mapping or focused studies

### Post-Association Filtering and Prioritization
- **LD-based signal refinement**:
  - Conditional analysis to identify independent signals
  - Stepwise regression for multi-allelic effects
  - LD clumping to obtain independent associated variants

- **Functional prioritization**:
  - Leverage epigenomic annotations (enhancers, promoters)
  - eQTL overlap analysis
  - Tissue-specific regulatory elements
  - Evolutionary conservation scores

- **Cross-phenotype comparisons**:
  - Pleiotropy with related traits
  - Genetic correlation (LDSC)
  - Mendelian Randomization for causal inference

## Interpretation Considerations

### Understanding Statistical Power
- **Power determinants**:
  - Sample size (cases and controls)
  - Effect size (odds ratio, beta)
  - Minor allele frequency
  - Imputation quality for non-genotyped variants

- **Power calculations**:
  - A priori power analysis to determine required sample size
  - Post-hoc analysis of minimum detectable effect size
  - Consider power differences across frequency spectrum

- **Sample size considerations**:
  - Modern GWAS often require >10,000 samples for common variants
  - Rare variant associations may require >50,000 samples
  - Consider meta-analysis for increased power

### Common Pitfalls and Solutions
- **Inflation due to uncorrected confounding**:
  - Always check QQ-plots and genomic inflation factor (λ)
  - Inflation patterns can distinguish polygenicity from confounding
  - LD Score regression to differentiate confounding from polygenic signal

- **Winner's curse**:
  - Effect size overestimation in discovery cohorts
  - Use shrinkage estimators or split-sample approaches
  - Critical for accurate genetic risk prediction

- **Interpretation of negative results**:
  - Consider power limitations rather than absence of effect
  - Focus on effect size confidence intervals
  - Meta-analysis with other studies if available

### Biological Interpretation Strategies
- **Functional annotation**:
  - Annotate variants with predicted functional impact
  - Consider non-coding regulatory effects
  - Determine tissue-specific effects

- **Pathway and network analysis**:
  - Gene-set enrichment analysis
  - Protein-protein interaction networks
  - Tissue-specific gene expression patterns

- **Integration with other omics data**:
  - eQTL/pQTL/meQTL data for functional validation
  - Chromatin conformation to identify gene targets
  - Single-cell data for cell type-specific effects

## Resources and References
- GWAS Catalog: https://www.ebi.ac.uk/gwas/
- PLINK Tutorial: https://zzz.bwh.harvard.edu/plink/
- UK Biobank GWAS Pipeline: https://github.com/Nealelab/UK_Biobank_GWAS
- SAIGE Documentation: https://github.com/weizhouUMICH/SAIGE
- LDSC Tutorial: https://github.com/bulik/ldsc/wiki/Tutorials
"""
        ),
        FewShotExample(
            query="How do I develop and validate polygenic risk scores from GWAS data?",
            context="The user has summary statistics from a large GWAS of heart disease and wants to develop a polygenic risk score for disease prediction.",
            response="""
# Developing and Validating Polygenic Risk Scores

## Approach Overview
Creating effective polygenic risk scores (PRS) involves:
1. Selecting and processing appropriate GWAS summary statistics
2. Choosing a PRS construction method
3. Tuning parameters and thresholds
4. Rigorous validation and evaluation
5. Interpreting and applying the scores appropriately

## Key Decision Points

### Input Data Preparation
- **Summary statistics quality assessment**:
  - Check for proper genomic build and strand alignment
  - Verify effect allele harmonization
  - Confirm sample overlap between discovery and target datasets
  - Examine inflation factors and potential biases

- **Target dataset requirements**:
  - Independent from discovery GWAS (no sample overlap)
  - Similar ancestry composition when possible
  - Complete phenotype information including relevant covariates
  - Properly QC'd genotype data (see GWAS QC standards)

- **Reference panel selection**:
  - Population-matched for LD information
  - Sufficient sample size (e.g., 1000 Genomes, HRC, TOPMed)
  - Consistent genomic build with summary statistics and target data

### PRS Construction Method Selection
- **Traditional approaches**:
  - P-value thresholding and LD pruning (P+T)
  - Simple weighted sum of allele counts × effect sizes
  - Range of p-value thresholds (5e-8, 1e-6, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.5, 1)

- **LD-aware methods**:
  - LDpred/LDpred2 (Bayesian approach with LD adjustment)
  - PRS-CS (continuous shrinkage priors)
  - lassosum (penalized regression approach)
  - SBayesR (mixture of Gaussians model)

- **Machine learning approaches**:
  - Penalized regression (elastic net, lasso)
  - Support vector machines for variant selection
  - Random forests for non-linear effects
  - Deep learning for complex architectures

- **Special considerations**:
  - Multi-ethnic PRS methods (PRS-CSx, XP-BLUP)
  - Functionally informed PRS (weighted by genomic annotations)
  - Pathway-specific sub-scores

### Parameter Tuning and Optimization
- **P+T parameter selection**:
  - Grid search across p-value thresholds
  - LD pruning parameters (r² thresholds of 0.1, 0.2, or 0.5)
  - Clumping window sizes (250kb, 500kb, 1Mb)

- **LD-aware parameter tuning**:
  - LDpred polygenicity parameter (proportion of causal variants)
  - PRS-CS global shrinkage parameter
  - lassosum regularization parameter (λ)

- **Cross-validation strategies**:
  - k-fold cross-validation within target dataset
  - Nested cross-validation for unbiased estimates
  - Hold-out validation sets

- **Objective functions**:
  - Area under ROC curve for binary outcomes
  - R² or explained variance for quantitative traits
  - Clinical net reclassification improvement (NRI)
  - Integrated discrimination improvement (IDI)

### Validation and Evaluation
- **Performance metrics**:
  - AUC/C-statistic for discrimination
  - Calibration plots for accuracy
  - Odds ratios per standard deviation of PRS
  - Positive/negative predictive values at different thresholds

- **Stratified analysis**:
  - Performance across ancestry groups
  - Sex-specific performance
  - Age-stratified evaluation
  - Comparison against family history

- **Comparison with existing models**:
  - Improvement over clinical risk scores
  - Additive value to established risk factors
  - Head-to-head comparison with published PRS

- **Survival analysis approaches**:
  - Hazard ratios and survival curves
  - Time-dependent AUC
  - Competing risk analysis where appropriate

## Interpretation Considerations

### Clinical Utility Assessment
- **Risk stratification approaches**:
  - Percentile-based risk categories (e.g., top 1%, 5%, 20%)
  - Absolute risk calculation using baseline incidence
  - Number needed to screen/test calculations
  - Age-specific risk trajectories

- **Intervention thresholds**:
  - Determination of clinically meaningful thresholds
  - Decision curve analysis
  - Cost-effectiveness considerations
  - Integration with clinical guidelines

- **Practical implementation factors**:
  - Laboratory and computational requirements
  - Interpretability for clinicians and patients
  - Integration with electronic health records
  - Regulatory and ethical considerations

### Common Challenges and Solutions
- **Transferability across populations**:
  - Reduced performance in non-European populations
  - Population-specific LD patterns affect portability
  - Consider transfer learning approaches
  - Develop population-specific or trans-ancestry PRS

- **Effect size bias correction**:
  - Winner's curse adjustment
  - Regression-to-the-mean effects
  - Empirical Bayes shrinkage methods
  - Summary statistics recalibration

- **Phenotype heterogeneity**:
  - Subtype-specific scores where appropriate
  - Consider age-at-onset stratification
  - Account for comorbidities
  - Recognize limitations for complex phenotype definitions

### Future Directions and Enhancements
- **Integration with other data types**:
  - Multi-omics PRS (combining with methylation, expression)
  - Inclusion of rare variant information
  - Incorporation of structural variants
  - Environmental interaction modeling (GxE)

- **Advanced methodological approaches**:
  - Bayesian networks for causal modeling
  - Pleiotropy-aware scores for related conditions
  - Transfer learning from large to small populations
  - Reinforcement learning for dynamic risk assessment

- **Emerging applications**:
  - Drug response prediction
  - Treatment stratification
  - Disease trajectory modeling
  - Early-life risk prediction

## Resources and References
- PRS-CS GitHub: https://github.com/getian107/PRScs
- LDpred2 Tutorial: https://privefl.github.io/bigsnpr/articles/LDpred2.html
- PGS Catalog: https://www.pgscatalog.org/
- PRSice-2: https://choishingwan.github.io/PRSice/
- PRS Tutorial: https://choishingwan.github.io/PRS-Tutorial/
"""
        ),
        FewShotExample(
            query="How do I perform fine-mapping and functional analysis to identify causal variants from GWAS results?",
            context="The user has identified several significant loci in their GWAS and wants to narrow down to the causal variants and understand their functional impact.",
            response="""
# Fine-Mapping and Functional Analysis of GWAS Loci

## Approach Overview
Effective fine-mapping and functional analysis involves:
1. Defining credible sets of potential causal variants
2. Integrating functional genomics data for prioritization
3. Validating function through computational and experimental approaches
4. Identifying target genes and biological mechanisms
5. Assessing relevance to disease biology

## Key Decision Points

### Fine-Mapping Strategy Selection
- **Statistical fine-mapping approaches**:
  - Bayesian methods (FINEMAP, CAVIARBF, PAINTOR)
  - Penalized regression approaches (LASSO, elastic net)
  - Approximate Bayes Factor methods
  - Multiple-causal variant methods vs. single-causal assumption

- **Input data requirements**:
  - Individual-level genotypes vs. summary statistics
  - Local LD information (reference panel selection)
  - Sample size and power considerations
  - Functional annotations for informative priors

- **Parameter selection**:
  - Number of causal variants per locus
  - Prior probabilities for causality
  - Credible set cutoffs (90% or 95% posterior probability)
  - LD window size definition (typically 500kb-1Mb)

- **Cross-population fine-mapping**:
  - Leveraging LD differences between populations
  - Trans-ethnic meta-analysis approaches
  - Population-specific effect heterogeneity assessment

### Functional Annotation Integration
- **Variant-level annotations**:
  - Coding consequences (missense, nonsense, synonymous)
  - Regulatory elements (promoters, enhancers, CTCF sites)
  - Evolutionary conservation scores
  - Predicted functional impact (CADD, FATHMM, SIFT, PolyPhen)
  
- **Tissue/cell-type specific information**:
  - Chromatin accessibility (ATAC-seq, DNase-seq)
  - Histone modifications (ChIP-seq)
  - Transcription factor binding sites
  - 3D chromatin structure (Hi-C, ChIA-PET)

- **Multi-omics data integration**:
  - Expression QTLs (eQTLs) across relevant tissues
  - Protein QTLs (pQTLs)
  - Methylation QTLs (meQTLs)
  - Splicing QTLs (sQTLs)

- **Functional fine-mapping approaches**:
  - PAINTOR (integrates functional annotations)
  - fGWAS (estimates enrichment of annotations)
  - RiVIERA (Bayesian model with tissue-specific epigenomics)
  - FOCUS (fine-mapping of causal gene sets)

### Target Gene Identification
- **Positional approaches**:
  - Nearest gene (simplistic but common)
  - Genes within fixed window (e.g., 500kb)
  - TAD (Topologically Associated Domain) boundaries

- **QTL-based approaches**:
  - Colocalization of GWAS and eQTL signals
  - Mendelian Randomization for causal inference
  - Multi-tissue eQTL analysis (e.g., GTEx)
  - Splicing effects in disease-relevant tissues

- **Chromatin interaction evidence**:
  - Hi-C/Capture Hi-C to identify promoter-enhancer contacts
  - ChIA-PET for protein-mediated interactions
  - ATAC-seq or H3K27ac HiChIP
  - 4C/5C for targeted region analysis

- **Integrative approaches**:
  - TWAS/PrediXcan (gene expression imputation)
  - SMR/HEIDI (Summary-based Mendelian Randomization)
  - FOCUS (probabilistic fine-mapping with eQTLs)
  - DEPICT (gene prioritization with coexpression)

### Functional Validation Approaches
- **Computational validation**:
  - Cross-phenotype associations
  - Evolutionary signatures of selection
  - Protein structure modeling
  - Molecular dynamics simulations

- **In vitro experimental approaches**:
  - CRISPR editing of variants
  - Luciferase reporter assays for enhancer activity
  - EMSA for protein-DNA binding
  - Massively parallel reporter assays (MPRAs)

- **Cell-based models**:
  - Disease-relevant cell types
  - iPSC-derived models
  - Organoids for tissue-specific effects
  - High-throughput screening approaches

- **In vivo models**:
  - Mouse models with humanized loci
  - CRISPR knockin/knockout
  - Tissue-specific and inducible systems
  - Physiological and molecular phenotyping

## Interpretation Considerations

### Biological Pathway Analysis
- **Enrichment approaches**:
  - Gene Ontology (GO) term enrichment
  - KEGG/Reactome pathway analysis
  - MSigDB gene set collections
  - Tissue-specific pathway analysis

- **Network-based methods**:
  - Protein-protein interaction networks
  - Co-expression networks
  - Regulatory networks
  - Bayesian networks for causal pathways

- **Prior knowledge integration**:
  - Literature-based relationship extraction
  - Drug-target databases
  - Disease-gene associations
  - Animal model phenotypes

- **Novel pathway discovery**:
  - Data-driven pathway identification
  - Clustering of functional profiles
  - Cell type-specific pathway activity
  - Temporal dynamics of pathway activation

### Translational Implications
- **Therapeutic target identification**:
  - Druggability assessment
  - Existing drug repurposing potential
  - Pathway intervention points
  - Biomarker development

- **Disease mechanism insights**:
  - Causal gene-phenotype relationships
  - Context-specific molecular mechanisms
  - Temporal disease progression effects
  - Environmental interaction mechanisms

- **Clinical relevance assessment**:
  - Phenotypic spectrum of implicated genes
  - Genotype-phenotype correlation
  - Disease subtype stratification
  - Comorbidity explanation

- **Personalized medicine applications**:
  - Pharmacogenomic implications
  - Treatment response prediction
  - Adverse effect risk assessment
  - Preventive intervention targeting

### Common Challenges and Solutions
- **False positive reduction**:
  - Statistical colocalization testing
  - Triangulation with orthogonal evidence
  - Replication in independent populations
  - Careful multiple testing correction

- **Pleiotropy vs. causality**:
  - Distinguish vertical from horizontal pleiotropy
  - Mendelian Randomization sensitivity analyses
  - Multi-trait fine-mapping
  - Causal mediation analysis

- **Tissue/context specificity**:
  - Appropriate tissue/cell type selection
  - Single-cell resolution where possible
  - Developmental stage consideration
  - Stimulation/perturbation responses

- **Technical artifacts**:
  - LD reference mismatch effects
  - Summary statistic quality control
  - Annotation version consistency
  - Population specificity of annotations

## Resources and References
- FINEMAP: http://www.christianbenner.com/
- PAINTOR: https://github.com/gkichaev/PAINTOR_V3.0
- Coloc Tutorial: https://chr1swallace.github.io/coloc/
- FUMA GWAS: https://fuma.ctglab.nl/
- HaploReg: https://pubs.broadinstitute.org/mammals/haploreg/haploreg.php
"""
        )
    ],
    references=[
        "Buniello A, et al. (2019). The NHGRI-EBI GWAS Catalog of published genome-wide association studies. Nucleic Acids Research.",
        "Manolio TA, et al. (2009). Finding the missing heritability of complex diseases. Nature.",
        "Visscher PM, et al. (2017). 10 Years of GWAS discovery: Biology, function, and translation. American Journal of Human Genetics.",
        "Schaid DJ, et al. (2018). From genome-wide associations to candidate causal variants by statistical fine-mapping. Nature Reviews Genetics.",
        "Choi SW, et al. (2020). Tutorial: a guide to performing polygenic risk score analyses. Nature Protocols.",
        "Wray NR, et al. (2021). From basic science to clinical application of polygenic risk scores. JAMA Psychiatry."
    ]
)

# Export the prompt for use in the package
if __name__ == "__main__":
    # Test the prompt with a sample query
    user_query = "How do I interpret GWAS results when I have hundreds of significant hits?"
    
    # Generate prompt
    prompt = gwas_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../gwas_prompt.json", "w") as f:
        f.write(gwas_prompt.to_json())

   # Load prompt template from JSON
    with open("../gwas_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt