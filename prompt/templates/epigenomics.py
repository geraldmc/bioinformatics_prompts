"""Module containing the epigenomics discipline prompt template."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from prompt.templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create an epigenomics prompt template
epigenomics_prompt = BioinformaticsPrompt(
    discipline="Epigenomics",
    description=(
        "Epigenomics is the study of the complete set of epigenetic modifications on the genetic material of a cell. "
        "These modifications, such as DNA methylation and histone modifications, do not change the DNA sequence "
        "but play critical roles in regulating gene expression. Epigenomic alterations are involved in various "
        "biological processes including development, aging, and disease pathogenesis. Advanced bioinformatics tools "
        "enable the mapping and interpretation of these complex modifications across the genome, providing insights "
        "into disease mechanisms and potential therapeutic targets."
    ),
    key_concepts=[
        "DNA methylation (5-mC, 5-hmC)",
        "Histone modifications (acetylation, methylation, phosphorylation)",
        "Chromatin accessibility and nucleosome positioning",
        "Enhancer and promoter regulation",
        "Long non-coding RNAs in epigenetic regulation",
        "Epigenetic inheritance and reprogramming",
        "3D chromatin structure and topologically associating domains (TADs)",
        "Cell type-specific epigenetic signatures",
        "CpG islands and shores",
        "Enhancer-promoter interactions"
    ],
    common_tools=[
        "Bismark - Bisulfite read mapping and methylation calling",
        "MACS2 - ChIP-seq and ATAC-seq peak calling",
        "deepTools - Visualization and analysis of deep-sequencing data",
        "methylKit - R package for DNA methylation analysis",
        "ChromHMM - Chromatin state discovery and characterization",
        "HOMER - Motif discovery and next-gen sequencing analysis",
        "Juicer/HiC-Pro - Hi-C data processing pipelines",
        "bsseq/DSS - Differential methylation analysis",
        "bedtools - Genome arithmetic and interval manipulation",
        "EnrichedHeatmap - Visualization of epigenomic signals"
    ],
    common_file_formats=[
        {"name": "FASTQ", "description": "Raw sequencing reads with quality scores"},
        {"name": "BAM/SAM", "description": "Alignment files for sequencing data"},
        {"name": "BED", "description": "Genome interval data (peaks, regions of interest)"},
        {"name": "bigWig/bedGraph", "description": "Genome coverage and signal tracks"},
        {"name": "methylation call files (.cov, .bedGraph)", "description": "Methylation status at individual cytosines"},
        {"name": "hic/cool/mcool", "description": "Formats for storing chromatin interaction data"},
        {"name": "narrowPeak/broadPeak", "description": "Peak calls from ChIP-seq or ATAC-seq"}
    ],
    examples=[
        FewShotExample(
            query="How do I analyze whole-genome bisulfite sequencing data to identify differentially methylated regions?",
            context="The user has paired-end WGBS data from 6 tumor samples and 6 matched normal tissues and wants to identify cancer-specific DNA methylation changes.",
            response="""
# Analyzing Whole-Genome Bisulfite Sequencing Data

## Approach Overview
Analyzing WGBS data to identify differentially methylated regions (DMRs) involves:
1. Quality control and preprocessing of bisulfite-converted reads
2. Alignment to a reference genome with bisulfite-aware mappers
3. Methylation calling at individual cytosines
4. Identification of differentially methylated regions
5. Biological interpretation and integration with other data types

## Key Decision Points

### Quality Control and Preprocessing
- **Bisulfite conversion efficiency assessment**:
  - Measure using spike-in controls or mitochondrial DNA
  - Expect >98% conversion efficiency
  - Poor conversion leads to methylation overestimation
  - Check for bimodal quality score distributions (common in bisulfite data)

- **Adapter and quality trimming considerations**:
  - More critical for WGBS due to lower complexity
  - Trim low-quality bases from 3' ends (Phred <20)
  - Remove adapters with tools like Trim Galore or Trimmomatic
  - Consider merging overlapping paired-end reads

- **Technical bias correction**:
  - M-bias plots to identify systematic errors by read position
  - GC bias assessment and normalization
  - Remove PCR duplicates to avoid methylation bias
  - Account for coverage biases across CpG density spectrum

### Alignment Strategy
- **Bisulfite-aware aligner selection**:
  - Bismark (widely used, good balance of accuracy and efficiency)
  - BSSeeker3 (faster but potentially less accurate)
  - BSMAP/GNUMAP-bs (implements different alignment algorithms)
  - BWA-meth (memory efficient, suitable for large genomes)

- **Reference genome preparation**:
  - C→T and G→A converted references required
  - Include spike-in sequences for conversion efficiency
  - Mask repetitive regions to improve mapping
  - Consider using masked or unmasked reference based on research question

- **Alignment parameters**:
  - Typically allow more mismatches than regular alignment
  - Adjust seed length (shorter for higher sensitivity)
  - Consider non-directional libraries if applicable
  - Use paired-end mode for improved mapping accuracy

- **Non-CpG methylation handling**:
  - Decide whether to analyze CHG and CHH contexts
  - May need separate pipeline for plants or specific tissues
  - Important for stem cells, neurons, and plant methylomes
  - Requires more stringent mapping due to lower prevalence

### Methylation Calling
- **Coverage thresholds**:
  - Minimum 5-10x coverage for reliable methylation calls
  - Consider higher thresholds (20-30x) for heterogeneous tissues
  - Balance between coverage requirement and genome-wide scope
  - Assess coverage distribution before setting thresholds

- **Methylation metrics**:
  - Beta values (0-1 scale, bimodal, more intuitive)
  - M-values (logit transformation, more statistically valid)
  - Coverage-weighted metrics for variable depth
  - Region-level aggregation methods

- **Smoothing considerations**:
  - Local smoothing leverages correlation between nearby CpGs
  - Kernel smoothing versus fixed window approaches
  - Statistical approaches like BSmooth
  - Trade-off between noise reduction and resolution

- **Single-CpG versus regional analysis**:
  - Individual CpG analysis has high noise but maximum resolution
  - Binned/window approaches reduce noise but lose resolution
  - Consider biological question when choosing granularity
  - Analyze both individual CpGs and regional patterns for comprehensive view

### Differential Methylation Analysis
- **Statistical model selection**:
  - Beta-binomial models (DSS, RADMeth)
  - Logistic regression with or without random effects
  - Bayesian approaches for improved handling of low coverage
  - Non-parametric methods for robustness

- **DMR calling strategies**:
  - Bump-hunting approaches (DMRcate, BSmooth)
  - Window-based methods with multiple testing correction
  - HMM-based segmentation (ComMet)
  - Region-level aggregation then testing

- **Covariate adjustment**:
  - Adjust for age and sex differences
  - Cell type heterogeneity correction critical
  - Batch effects and technical covariates
  - Consider genetic background when applicable

- **Multiple testing correction**:
  - FDR control for DMR discovery
  - Permutation-based approaches for regional testing
  - Spatial adjustment for correlation between CpGs
  - q-value or Benjamini-Hochberg procedures

### Biological Interpretation
- **Genomic context annotation**:
  - Annotate with respect to genes, promoters, enhancers
  - CpG island context (islands, shores, shelves, open sea)
  - Chromatin state overlaps
  - Transcription factor binding sites

- **Integration with other omics data**:
  - Correlate with gene expression data
  - Overlay with histone modifications
  - Connect to chromatin accessibility
  - Combine with genetic variation information

- **Pathway and network analysis**:
  - Gene set enrichment for genes near DMRs
  - Epigenetic regulator network analysis
  - DNA methylation modules identification
  - Tissue-specific regulatory network integration

- **Visualization approaches**:
  - Genome browser tracks for locus-specific inspection
  - Heatmaps for clustering and pattern discovery
  - Circular plots for genome-wide overview
  - Combine with annotation tracks for context

## Interpretation Considerations

### Cancer-Specific Considerations
- **Global hypomethylation patterns**:
  - Widespread loss of methylation in repetitive elements
  - Chromosomal instability associations
  - Reactivation of endogenous retroviruses
  - Measuring global levels versus focal changes

- **Focal hypermethylation analysis**:
  - CpG island promoter methylation
  - Tumor suppressor silencing
  - Polycomb target gene enrichments
  - Cancer type-specific patterns

- **Copy number variation interference**:
  - Adjust methylation calls for copy number changes
  - Recognize that deletion/amplification alters apparent methylation
  - Joint segmentation of methylation and copy number
  - Allele-specific methylation in regions with LOH

- **Tumor heterogeneity challenges**:
  - Deconvolution of cell type mixtures
  - Clonal versus subclonal methylation changes
  - Single-cell approaches when applicable
  - Paired analysis with tumor microdissection

### Common Challenges and Solutions
- **Coverage variability**:
  - Weight statistical tests by coverage
  - Apply minimum coverage thresholds
  - Imputation for low-coverage regions
  - Bin regions for more uniform coverage

- **Bisulfite conversion artifacts**:
  - Incomplete conversion false positives
  - Over-conversion false negatives
  - Non-CpG filtering to assess conversion
  - Spike-in controls for calibration

- **Strand specificity**:
  - Merge cytosines from both strands for CpG sites
  - Separate analysis for non-CpG methylation by strand
  - Directional versus non-directional library protocols
  - Assymetric methylation pattern detection

- **Repetitive element mapping**:
  - Special handling for evolutionarily young repeats
  - SLIM approach for improved repeat mapping
  - Consider alignment quality scores
  - Balance between uniqueness and coverage

### Integration with Clinical Information
- **Biomarker development**:
  - Identify robust, reproducible methylation markers
  - Validation in independent cohorts
  - Technical validation with orthogonal methods
  - Clinical outcome correlation

- **Therapeutic relevance**:
  - Identify actionable epigenetic alterations
  - DNA methyltransferase inhibitor response prediction
  - Epigenetic sensitization to other therapies
  - Biomarkers for immunotherapy response

- **Prognostic classification**:
  - Methylation-based tumor classification
  - Survival prediction models
  - Integration with genetic and clinical factors
  - Multivariate versus methylation-specific models

## Resources and References
- Bismark Tutorial: https://github.com/FelixKrueger/Bismark/tree/master/Docs
- DSS Bioconductor Package: https://bioconductor.org/packages/release/bioc/html/DSS.html
- methylKit Documentation: https://bioconductor.org/packages/release/bioc/html/methylKit.html
- ENCODE DNA Methylation Standards: https://www.encodeproject.org/data-standards/wgbs/
- Roadmap Epigenomics Visualizations: http://epigenomegateway.wustl.edu/
"""
        ),
        FewShotExample(
            query="How do I analyze ATAC-seq data to identify differential chromatin accessibility between conditions?",
            context="The user has ATAC-seq data from treated and untreated stem cell samples (3 replicates per condition) and wants to identify treatment-induced changes in chromatin accessibility.",
            response="""
# ATAC-seq Analysis for Differential Chromatin Accessibility

## Approach Overview
Analyzing ATAC-seq data to identify differential accessibility involves:
1. Quality control and preprocessing of sequencing reads
2. Alignment to a reference genome
3. Peak calling to identify accessible chromatin regions
4. Differential accessibility analysis between conditions
5. Functional interpretation and integration with other data types

## Key Decision Points

### Quality Control and Preprocessing
- **ATAC-specific quality metrics**:
  - Fragment size distribution (nucleosome-free, mono-, di-, tri-nucleosome patterns)
  - TSS enrichment score (>6 for high-quality samples)
  - FRiP (Fraction of Reads in Peaks) (>0.3 desirable)
  - Library complexity and PCR duplication rate
  - Mitochondrial read percentage (often high in ATAC-seq)

- **Read preprocessing considerations**:
  - Adapter trimming (Nextera adapters for standard ATAC-seq)
  - Quality filtering (Phred score >20 typically)
  - Mitochondrial read removal (can consume >50% of reads)
  - Read length considerations (paired-end preferred, at least 50bp)

- **Technical bias assessment**:
  - GC bias evaluation
  - Tn5 insertion bias (distinct from other methods)
  - Batch effects and technical variation
  - Sample-to-sample correlation checks

### Alignment Strategy
- **Aligner selection**:
  - Bowtie2 (most common for ATAC-seq)
  - BWA-MEM (alternative with slightly different characteristics)
  - STAR (if spliced alignments are desired)
  - Minimap2 (for longer reads)

- **Alignment parameters**:
  - Proper paired-end alignment
  - Adjust for soft-clipping at read ends
  - Filter for high mapping quality (MAPQ >20-30)
  - Unique mapping versus multi-mapped reads

- **Post-alignment processing**:
  - Duplicate removal (crucial for ATAC-seq)
  - Shifting aligned reads (+4bp for forward strand, -5bp for reverse)
  - Blacklist region filtering
  - Chromosome selection (typically exclude chrM, random contigs)

- **Fragment selection**:
  - Nucleosome-free regions (<100bp)
  - Mono-nucleosomal fragments (~180-250bp)
  - All fragments versus fragment size selection
  - Implications for different biological questions

### Peak Calling Strategy
- **Peak caller selection**:
  - MACS2 (most widely used, designed for punctate signal)
  - Genrich (ATAC-seq specific features)
  - HMMRATAC (designed for ATAC-seq signal structure)
  - F-seq (density estimation approach)

- **Peak calling parameters**:
  - Shift size for Tn5 insertion site (typically 100bp for MACS2)
  - q-value/FDR threshold (typically 0.01 or 0.05)
  - Peak width constraints
  - Local background estimation method

- **Peak set handling**:
  - Sample-specific peaks versus merged consensus peaks
  - Irreproducible Discovery Rate (IDR) for reproducibility
  - Fixed-width peak approaches
  - Broad versus narrow accessibility domains

- **Control library considerations**:
  - IgG controls versus input DNA
  - Naked DNA with Tn5 treatment
  - Computational background models
  - No-control approach with local background

### Differential Accessibility Analysis
- **Quantification approach**:
  - Read counting in peak regions
  - Signal track generation for visualization
  - Count normalization methods (CPM, RPKM, TMM)
  - Signal versus count-based comparisons

- **Statistical testing methods**:
  - DESeq2/edgeR for count-based analysis
  - limma-voom for improved sensitivity
  - Specialized tools like DiffBind or chromVAR
  - Non-parametric approaches for robustness

- **Multiple testing correction**:
  - FDR control for peak-level comparisons
  - Prior weighting based on peak strength
  - Spatial correlation adjustment
  - Permutation-based approaches

- **Normalization considerations**:
  - Library size differences
  - Global accessibility shifts
  - Spike-in normalization for absolute changes
  - Housekeeping region normalization

### Functional Interpretation
- **Motif analysis**:
  - De novo motif discovery (MEME, HOMER)
  - Known motif enrichment
  - Differential motif activity (HINT-ATAC, chromVAR)
  - Footprinting analysis for TF binding evidence

- **Genomic annotation**:
  - Promoter, enhancer, boundary elements
  - Gene associations and distance weighting
  - Chromatin state overlaps
  - Evolutionary conservation

- **Gene-level integration**:
  - Correlation with gene expression
  - Assignment of regulatory elements to genes
  - Enhancer-promoter interactions (when Hi-C available)
  - Pathway and network analysis

- **Visualization approaches**:
  - Genome browser tracks with paired conditions
  - Aggregated signal at features (TSS, enhancers)
  - Heatmaps for pattern discovery
  - Principal component or t-SNE plots

## Interpretation Considerations

### Stem Cell Specific Considerations
- **Pluripotency regulatory network**:
  - Focus on OCT4, SOX2, NANOG binding regions
  - Bivalent domain dynamics
  - Lineage-specific enhancer priming
  - Developmental gene poising

- **Differentiation dynamics**:
  - Sequential chromatin remodeling during lineage commitment
  - Pioneer factor activity
  - Silencing versus activation patterns
  - Heterogeneity in differentiation stage

- **Epigenetic reprogramming effects**:
  - Treatment effects on pluripotency maintenance
  - Cell state transitions and intermediates
  - Relationship to histone modifications
  - DNA methylation interactions

- **Technical challenges in stem cells**:
  - Higher background in pluripotent cells
  - Heterogeneity effects on signal
  - Appropriate control cell populations
  - Cell cycle effects on accessibility

### Common Challenges and Solutions
- **Signal to noise optimization**:
  - Optimizing cell numbers and lysis conditions
  - Improved Tn5 reaction conditions
  - Deeper sequencing for low-input samples
  - Fragment size selection strategies

- **Cell type heterogeneity**:
  - Single-cell ATAC-seq when feasible
  - Deconvolution approaches
  - Marker region analysis for purity
  - Integration with scRNA-seq data

- **Replicate variability**:
  - Batch correction methods
  - Consensus peak identification
  - Irreproducible Discovery Rate framework
  - Quantitative quality metrics for exclusion decisions

- **Technical biases**:
  - Tn5 sequence preference correction
  - GC normalization methods
  - Mappability bias awareness
  - Transposase concentration consistency

### Functional Validation Approaches
- **Experimental validation methods**:
  - CRISPR deletion/modification of regions
  - Reporter assays for enhancer function
  - Targeted epigenome editing
  - 4C/5C/Hi-C for interaction validation

- **Target gene validation**:
  - Gene expression correlation
  - eQTL overlap evidence
  - Perturbation followed by expression analysis
  - Allele-specific accessibility and expression

- **TF binding validation**:
  - Footprinting depth and shapes
  - ChIP-seq confirmation
  - Motif mutation effects
  - TF perturbation effects on accessibility

- **Orthogonal method comparison**:
  - DNase-seq concordance
  - MNase-seq nucleosome positioning
  - HiChIP/PLAC-seq interaction validation
  - CUT&RUN/CUT&Tag for TF binding

## Resources and References
- ENCODE ATAC-seq Standards: https://www.encodeproject.org/atac-seq/
- ATACseqQC Bioconductor Package: https://bioconductor.org/packages/release/bioc/html/ATACseqQC.html
- MACS2 GitHub: https://github.com/macs3-project/MACS
- DiffBind Tutorial: https://bioconductor.org/packages/release/bioc/vignettes/DiffBind/inst/doc/DiffBind.pdf
- Greenleaf Lab ATAC-seq Resources: https://github.com/GreenleafLab/ATAC-seq-resources
"""
        ),
        FewShotExample(
            query="How can I integrate ChIP-seq and RNA-seq data to identify functional targets of a transcription factor?",
            context="The user has ChIP-seq for a transcription factor and RNA-seq data from wild-type and knockout cells for the same factor.",
            response="""
# Integrating ChIP-seq and RNA-seq for Transcription Factor Analysis

## Approach Overview
Integrating ChIP-seq and RNA-seq to identify functional TF targets involves:
1. Processing and analyzing each data type separately
2. Linking ChIP-seq binding sites to potential target genes
3. Determining differentially expressed genes due to TF knockout
4. Integrating binding and expression data to identify direct targets
5. Characterizing regulatory mechanisms and network effects

## Key Decision Points

### ChIP-seq Analysis Approach
- **Peak calling strategy**:
  - MACS2/MACS3 (most common, good for sharp peaks)
  - GEM (motif-aware peak calling)
  - SEACR (designed for CUT&RUN/CUT&Tag)
  - Signal threshold selection (q-value 0.01-0.05 typical)

- **Control selection**:
  - Input DNA (preferred, accounts for chromatin biases)
  - IgG control (accounts for antibody biases)
  - Knockout/knockdown sample (for TF specificity)
  - Control normalization methods

- **Binding site characteristics**:
  - Peak width distribution analysis
  - Signal strength and shape assessment
  - Motif enrichment verification
  - Cross-validation with external datasets

- **Quality assessment metrics**:
  - Fraction of reads in peaks (>1% minimum, >10% good)
  - Library complexity metrics
  - Strand cross-correlation
  - Peak number and characteristics versus published data

### RNA-seq Analysis Approach
- **Differential expression strategy**:
  - DESeq2 (robust, handles outliers well)
  - edgeR (good for experiments with few replicates)
  - limma-voom (good control of false positives)
  - Sleuth (for transcript-level analysis)

- **Expression preprocessing considerations**:
  - Read quality filtering
  - Appropriate normalization (TPM, FPKM, counts)
  - Batch effect correction
  - Sample outlier identification

- **Statistical threshold selection**:
  - FDR cutoff (typically 0.05, 0.01, or 0.1)
  - Fold change thresholds (1.5-2x common)
  - Expression level filters
  - Multiple testing correction method

- **RNA-seq analysis granularity**:
  - Gene-level versus transcript-level
  - Splicing and isoform analysis
  - Non-coding RNA consideration
  - Alternative promoter usage

### Binding Site to Gene Assignment
- **Distance-based approaches**:
  - Nearest TSS assignment
  - Fixed-window approach (e.g., ±50kb from TSS)
  - Weighted by distance (exponential decay models)
  - Bidirectional versus directional assignment

- **Topological domain consideration**:
  - TAD-constrained assignments
  - Insulator element boundaries
  - CTCF site constraints
  - Enhancer-promoter loops from Hi-C

- **Enhancer-promoter mapping**:
  - Correlation-based approaches
  - Physical interaction data integration
  - Activity-by-contact models
  - Expression quantitative trait loci (eQTLs)

- **Regulatory element classification**:
  - Promoter-proximal versus distal
  - Enhancer/silencer distinction
  - Tissue-specific versus constitutive
  - Active/poised/repressed states

### Integration Strategy
- **Direct overlap approaches**:
  - Intersect ChIP-seq peaks with DE gene promoters
  - Calculate enrichment statistics for overlap
  - Fisher's exact test for significance
  - Odds ratio for effect size

- **Correlation-based methods**:
  - Binding strength versus expression change
  - Binding proximity versus expression effect
  - Motif strength versus regulatory impact
  - Multi-factor models for prediction

- **Causality assessment**:
  - Directionality of effect (activation/repression)
  - Time-course data for temporal relationships
  - Dose-dependent responses
  - Mediator/cofactor dependencies

- **Network-based integration**:
  - Graph-based models of regulation
  - Target gene interaction networks
  - TF-TF regulatory circuits
  - Feed-forward and feedback loops identification

### Validation and Refinement
- **Motif analysis**:
  - De novo motif discovery at binding sites
  - Motif strength correlation with expression changes
  - Secondary motif discovery for cofactors
  - Motif spacing and orientation analysis

- **Functional genomics integration**:
  - Open chromatin correlation (ATAC-seq, DNase-seq)
  - Histone modification patterns
  - 3D chromatin organization
  - Chromatin remodeler binding

- **Experimental validation approaches**:
  - Reporter assays for enhancer function
  - CRISPR perturbation of binding sites
  - Single-cell approaches for heterogeneity
  - Mass spectrometry for protein partners

- **Computational validation**:
  - Cross-species conservation analysis
  - Independent dataset replication
  - Alternative statistical approaches
  - Feature importance modeling

## Interpretation Considerations

### Direct versus Indirect Effects
- **Identifying direct targets**:
  - Binding + differential expression
  - Immediate early response genes
  - Consistent direction of regulation
  - Motif presence at binding sites

- **Recognizing indirect effects**:
  - Expression change without binding
  - Delayed response genes
  - Secondary TF involvement
  - Network propagation effects

- **Distinguishing activation from repression**:
  - Upregulation versus downregulation in knockout
  - Context-dependent dual roles
  - Co-activator/co-repressor interactions
  - Enhancer versus silencer function

- **Interpreting binding without expression change**:
  - Poised or reserved regulation
  - Condition-specific activity
  - Redundant regulation
  - Non-functional binding

### Biological Context Integration
- **Cell type specificity**:
  - Pioneer factor versus settler TF roles
  - Tissue-specific regulatory networks
  - Developmental stage considerations
  - Environmental response differences

- **Chromatin state relevance**:
  - Active versus repressed chromatin
  - Accessibility prerequisites
  - Bivalent domain dynamics
  - Enhancer activation states

- **Pathway and network implications**:
  - Enriched biological pathways among targets
  - Regulatory network hierarchy
  - Master regulator assessment
  - Signaling pathway integration

- **Evolutionary considerations**:
  - Binding site conservation
  - Target gene set conservation
  - Regulatory circuit persistence
  - Species-specific adaptations

### Common Challenges and Solutions
- **Antibody quality issues**:
  - Validation with knockout controls
  - Comparison with published datasets
  - Motif enrichment confirmation
  - Orthogonal methods (CUT&RUN)

- **Distinguishing functional binding**:
  - Integration with accessibility data
  - Cofactor binding overlap
  - Regulatory element annotations
  - Massively parallel reporter assays

- **Connecting distal elements to genes**:
  - Chromatin conformation data
  - Activity correlation across samples
  - CRISPR perturbation screens
  - eQTL evidence

- **Sample heterogeneity effects**:
  - Single-cell approaches when feasible
  - Deconvolution methods
  - Cell sorting for purification
  - Marker gene assessment

### Emerging Approaches
- **Single-cell multi-omics**:
  - scRNA-seq with scATAC-seq
  - Multi-modal data integration
  - Trajectory analysis for dynamics
  - Heterogeneity in TF function

- **TF footprinting methods**:
  - High-resolution binding detection
  - Protein-DNA interaction details
  - Cooperative binding evidence
  - Nucleosome positioning relationships

- **Deep learning integration**:
  - Sequence-based TF binding prediction
  - Enhancer-gene assignment models
  - Expression effect prediction
  - Multi-modal data integration

- **Spatial transcriptomics**:
  - Tissue context for regulation
  - Spatial regulatory domains
  - Cell-cell interaction effects
  - Niche-specific TF function

## Resources and References
- ChIP-seq Guidelines: https://www.encodeproject.org/chip-seq/transcription_factor/
- RNA-seq Analysis Workflow: https://bioconductor.org/packages/release/workflows/html/rnaseqGene.html
- ChIPseeker: https://bioconductor.org/packages/release/bioc/html/ChIPseeker.html
- ENCODE ChIP-seq Resources: https://www.encodeproject.org/chip-seq/
- MEME Suite Tools: https://meme-suite.org/
"""
        )
    ],
    references=[
        "Roadmap Epigenomics Consortium, et al. (2015). Integrative analysis of 111 reference human epigenomes. Nature.",
        "Krueger F, Andrews SR. (2011). Bismark: a flexible aligner and methylation caller for Bisulfite-Seq applications. Bioinformatics.",
        "Buenrostro JD, et al. (2015). ATAC-seq: A Method for Assaying Chromatin Accessibility Genome-Wide. Current Protocols in Molecular Biology.",
        "Corces MR, et al. (2018). The chromatin accessibility landscape of primary human cancers. Science.",
        "Luo C, et al. (2018). Single nucleus multi-omics links human cortical cell regulatory genome diversity to disease risk variants. Cell.",
        "Zhu H, et al. (2016). Computational analysis of transcription factor binding sites across diverse cell types. Genome Research."
    ]
)

# Export the prompt for use in the package
if __name__ == "__main__":
    # Test the prompt with a sample query
    user_query = "How do I analyze histone modification ChIP-seq data to identify cell-type specific enhancers?"
    
    # Generate prompt
    prompt = epigenomics_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("epigenomics_prompt.json", "w") as f:
        f.write(epigenomics_prompt.to_json())