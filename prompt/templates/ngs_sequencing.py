import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create an NGS sequencing prompt template
ngs_sequencing_prompt = BioinformaticsPrompt(
    discipline="Next-Generation Sequencing Analysis",
    description=(
        "Next-Generation Sequencing (NGS) analysis encompasses the computational methods and frameworks "
        "used to process, analyze, and interpret high-throughput sequencing data. Modern NGS platforms "
        "generate massive volumes of short or long reads that require specialized bioinformatics approaches "
        "for quality control, alignment, assembly, variant detection, and functional interpretation. "
        "As sequencing technologies continue to evolve—from short-read technologies like Illumina to "
        "long-read platforms such as Oxford Nanopore and PacBio—bioinformatics pipelines must adapt to "
        "different error profiles, read lengths, and applications ranging from whole genome sequencing "
        "to targeted panels, transcriptomics, and epigenomics."
    ),
    key_concepts=[
        "Sequencing platform selection and technology tradeoffs",
        "Read quality assessment and preprocessing strategies",
        "Read alignment and mapping approaches",
        "De novo assembly algorithms and validation",
        "Variant calling and filtering frameworks",
        "Structural variant detection",
        "Sequencing depth and coverage considerations",
        "Batch effect management and study design"
    ],
    common_tools=[
        "FastQC/MultiQC - Quality control assessment",
        "BWA/Bowtie2 - Short read alignment",
        "STAR/HISAT2 - RNA-seq alignment",
        "Minimap2/NGMLR - Long read alignment",
        "GATK - Variant discovery and genotyping",
        "SPAdes/MEGAHIT - De novo assembly",
        "Canu/Flye - Long read assembly",
        "Salmon/Kallisto - Transcript quantification"
    ],
    common_file_formats=[
        {"name": "FASTQ", "description": "Raw sequencing reads with quality scores"},
        {"name": "BAM/CRAM", "description": "Aligned sequencing reads"},
        {"name": "VCF/gVCF", "description": "Variant calls and genotypes"},
        {"name": "BED/GFF/GTF", "description": "Genomic feature annotations"},
        {"name": "FASTA", "description": "Sequence data without quality scores"},
        {"name": "BCF", "description": "Binary variant call format"},
        {"name": "bigWig/bedGraph", "description": "Genomic data visualization formats"}
    ],
    examples=[
        FewShotExample(
            query="What are the key considerations when designing an NGS study for rare variant discovery in a complex disease?",
            context="The user is planning a whole genome or exome sequencing study to identify rare variants associated with a complex disease and needs guidance on study design decisions.",
            response="""
# NGS Study Design for Rare Variant Discovery

## Conceptual Framework

Rare variant discovery in complex diseases requires balancing multiple factors:

1. **Sequencing depth and breadth**: Coverage strategies for variant detection
2. **Sample selection and size**: Cohort composition and statistical power
3. **Sequencing approach**: Technology and targeting strategy selection
4. **Analysis strategy**: Variant calling, filtering, and interpretation pipeline
5. **Validation framework**: Confirmation and replication approaches

Each decision impacts discovery power, false discovery rates, and resource utilization.

## Decision Framework

### Cohort Design Decisions
- **Sample selection strategy**:
  - **Family-based designs**:
    - Trio/quad sequencing for de novo variant discovery
    - Extended pedigrees for co-segregation analysis
    - Affected-only sequencing in distantly related individuals
  
  - **Case-control designs**:
    - Matched vs. unmatched controls
    - Population-specific vs. multi-ancestry approach
    - Extreme phenotype selection
    - Quantitative trait stratification
  
  - **Hybrid approaches**:
    - Two-stage family and population designs
    - Isolated population advantages
    - Founder population considerations

- **Sample size determination**:
  - Variant frequency considerations
  - Effect size assumptions
  - Desired statistical power
  - Multiple testing burden
  - Resource constraints vs. discovery potential

- **Phenotyping depth strategy**:
  - Deep phenotyping advantages
  - Endophenotype selection
  - Longitudinal vs. cross-sectional assessment
  - Environmental exposure documentation
  - Biomarker integration

### Sequencing Strategy Selection
- **Sequencing approach**:
  - **Whole genome sequencing**:
    - Advantages: Complete coverage, structural variant detection, regulatory regions
    - Limitations: Cost, analysis complexity, incidental findings
  
  - **Whole exome sequencing**:
    - Advantages: Focused on protein-coding regions, established pipelines, cost-efficiency
    - Limitations: Missed regulatory variants, capture biases, uneven coverage
  
  - **Custom panels**:
    - Advantages: Higher depth, focused analysis, cost-effective
    - Limitations: Missing novel genes, limited discovery potential

- **Platform selection considerations**:
  - Short-read technologies (Illumina): High accuracy, established pipelines
  - Long-read technologies (PacBio/Nanopore): Better for structural variants, repetitive regions
  - Hybrid approaches: Combining technologies for comprehensive detection
  - Cost vs. data output tradeoffs

- **Sequencing depth determination**:
  - Coverage requirements for reliable rare variant calling (30-60x for WGS)
  - Depth variation across genomic regions
  - Sequencing technology error profiles
  - Mosaic variant detection needs
  - Cost vs. sensitivity tradeoffs

### Technical Implementation Planning
- **Sample preparation decisions**:
  - DNA extraction method standardization
  - Input quality/quantity requirements
  - PCR-free vs. amplification-based libraries
  - Exome capture kit selection (if applicable)
  - Batch processing strategy

- **Sequencing parameters**:
  - Read length optimization
  - Paired-end vs. single-end sequencing
  - Insert size selection
  - Multiplexing strategy
  - Lane allocation and flow cell planning

- **Quality control framework**:
  - Sample identity verification methods
  - Cross-contamination screening
  - Sequencing run QC metrics
  - Batch effect monitoring strategy
  - Positive/negative controls inclusion

### Analytical Pipeline Design
- **Primary analysis approach**:
  - Read alignment strategy selection
  - Quality score recalibration decisions
  - Duplicate marking protocols
  - Reference genome selection (GRCh38 vs. others)
  - Base quality thresholds

- **Variant calling strategy**:
  - Single-sample vs. joint calling approaches
  - SNV/indel caller selection and parameters
  - Structural variant detection methods
  - Caller ensemble approaches
  - Validation site integration

- **Filtering and prioritization framework**:
  - Quality-based filtering strategy
  - Allele frequency thresholds
  - Annotation sources selection
  - In silico prediction tool integration
  - Inheritance pattern modeling
  - Polygenic risk score consideration

### Validation and Interpretation Strategy
- **Technical validation approach**:
  - Orthogonal technology selection
  - Validation sample selection strategy
  - False positive assessment framework
  - Variant calling concordance evaluation
  - Replication in independent cohorts

- **Functional validation planning**:
  - Candidate variant prioritization strategy
  - Model system selection (cell lines, organisms)
  - Functional assay design considerations
  - Multi-omics data integration
  - Collaboration network development

- **Data sharing and integration**:
  - Database submission planning
  - Controlled access framework
  - Integration with existing resources
  - Collaborative analysis possibilities
  - Long-term data management

## Implementation Considerations

- **Start with clear hypotheses**: Define specific research questions before designing study
- **Consider a pilot phase**: Test workflow with subset of samples before full deployment
- **Plan for computational resources**: Ensure adequate storage and processing capacity
- **Develop concrete analysis plan**: Document statistical approaches before data generation
- **Consider ethical implications**: Plan for return of results and incidental findings

## Useful Resources

- [NHGRI Genome Sequencing Program](https://www.genome.gov/Funded-Programs-Projects/NHGRI-Genome-Sequencing-Program)
- [Genomics England Protocol](https://www.genomicsengland.co.uk/about-genomics-england/how-we-work/protocols)
- [CCDG Best Practices](https://www.genome.gov/Funded-Programs-Projects/NHGRI-Genome-Sequencing-Program/Centers-for-Common-Disease-Genomics)
- [gnomAD Browser](https://gnomad.broadinstitute.org/) - Population frequency resource
- [ClinGen Resource](https://clinicalgenome.org/) - Clinical variant interpretation
"""
        ),
        FewShotExample(
            query="How should I select and optimize an RNA-seq analysis pipeline for differential expression analysis?",
            context="The user is conducting a transcriptomics experiment and needs guidance on pipeline selection and optimization for differential gene expression analysis.",
            response="""
# RNA-seq Pipeline Selection and Optimization

## Conceptual Framework

Effective RNA-seq analysis pipeline selection requires balancing:

1. **Experimental design compatibility**: Alignment with study goals and design
2. **Technical specifications**: Read type, length, and depth considerations
3. **Analytical objectives**: Primary and secondary analysis goals
4. **Resource constraints**: Computational and time limitations
5. **Validation approach**: Quality assessment and result confirmation strategy

An optimal pipeline maximizes biological insight while minimizing technical artifacts.

## Decision Framework

### Experimental Context Assessment
- **Study objective alignment**:
  - **Primary analysis goals**:
    - Differential expression analysis
    - Alternative splicing detection
    - Fusion gene discovery
    - Novel transcript identification
    - Small RNA analysis
  
  - **Sample characteristics**:
    - Well-annotated vs. non-model organism
    - Expected transcriptome complexity
    - Tissue heterogeneity considerations
    - Known biological variance
    - Clinical vs. experimental source
  
  - **Experimental design features**:
    - Replicate number and type
    - Batch structure and confounding factors
    - Time course vs. endpoint design
    - Paired vs. unpaired comparisons
    - Multi-factor experimental structure

- **Library preparation impacts**:
  - Poly-A selection vs. ribosomal depletion effects
  - Stranded vs. unstranded protocol implications
  - UMI utilization for quantification
  - 3' bias considerations
  - Library complexity assessment

- **Sequencing specifications**:
  - Read length implications (50bp vs. 75bp vs. 150bp)
  - Paired-end vs. single-end design tradeoffs
  - Sequencing depth adequacy
  - Quality profile and error patterns
  - Multiplexing scheme impacts

### Pipeline Component Selection
- **Quality control and preprocessing**:
  - **QC assessment tools**:
    - FastQC/MultiQC for quality metrics
    - RSeQC for RNA-specific metrics
    - Quality threshold determination
  
  - **Preprocessing decisions**:
    - Adapter trimming necessity
    - Quality-based filtering strategy
    - rRNA/globin read removal approach
    - UMI handling (if applicable)
    - FASTQ interleaving/deinterleaving needs

- **Read mapping strategy**:
  - **Alignment approach selection**:
    - Splice-aware aligners (STAR, HISAT2)
    - Pseudo-alignment (Salmon, Kallisto)
    - Alignment-free methods tradeoffs
    - Genome vs. transcriptome alignment
  
  - **Reference selection**:
    - Genome build currency
    - Transcriptome annotation completeness
    - Inclusion of non-canonical transcripts
    - Addition of pathogen/contaminant sequences
    - Custom reference modifications

- **Quantification method**:
  - Count-based approaches (featureCounts, HTSeq)
  - Transcript-level quantification (Salmon, Kallisto)
  - Transcript vs. gene-level analysis
  - Multi-mapping read handling strategy
  - Normalization method selection

### Differential Analysis Framework
- **Statistical model selection**:
  - **Package selection considerations**:
    - DESeq2: Robust for small sample sizes
    - edgeR: Flexible dispersion modeling
    - limma-voom: Linear modeling capabilities
    - sleuth: Transcript-level uncertainty modeling
  
  - **Model specification**:
    - Simple vs. multi-factor designs
    - Nested vs. crossed factors
    - Continuous covariate inclusion
    - Interaction term modeling
    - Repeated measures handling

- **Data transformation decisions**:
  - Raw counts vs. TPM/FPKM/CPM
  - Variance stabilizing transformations
  - Log transformation approach
  - Batch correction methods
  - Outlier detection and handling

- **Multiple testing correction**:
  - FDR vs. FWER approaches
  - Significance threshold determination
  - Independent filtering implementation
  - Effect size thresholding
  - Visualization of significance

### Downstream Analysis Planning
- **Functional enrichment approach**:
  - GO/pathway analysis method selection
  - Gene set enrichment analysis
  - Network analysis integration
  - Cell type deconvolution needs
  - Multi-omics data integration

- **Visualization strategy**:
  - Standard plots (volcano, MA, heatmap)
  - Sample relationship visualization (PCA, MDS)
  - Pathway visualization approaches
  - Interactive visualization needs
  - Publication-quality figure generation

- **Result validation planning**:
  - Technical validation (qPCR targets)
  - Biological validation approach
  - Integration with external datasets
  - Comparison with literature findings
  - Independent cohort verification

### Implementation Optimization
- **Computational resource allocation**:
  - Memory requirements assessment
  - CPU core optimization
  - Disk space planning
  - Parallelization opportunities
  - Cloud vs. local computation

- **Workflow management**:
  - Pipeline orchestration tools
  - Containerization approach
  - Parameter optimization strategy
  - Checkpoint and restart capabilities
  - Version control implementation

- **Documentation and reproducibility**:
  - Parameter recording methodology
  - Environment capture approach
  - Results provenance tracking
  - Analysis reporting format
  - Code sharing strategy

## Implementation Considerations

- **Start with benchmarked pipelines**: Use established workflows as starting points
- **Run pilot analysis**: Test with subset of samples before full dataset
- **Implement quality checkpoints**: Verify data quality at each analytical stage
- **Consider positive controls**: Include samples with expected differences
- **Maintain flexible parameters**: Allow customization for specific sample types

## Useful Resources

- [RNA-seq workflow (Bioconductor)](https://bioconductor.org/packages/release/workflows/html/rnaseqGene.html)
- [ENCODE RNA-seq Standards](https://www.encodeproject.org/documents/cede0cbe-d324-4ce7-ace4-f0c3eddf5972/@@download/attachment/ENCODE%20Best%20Practices%20for%20RNA_v2.pdf)
- [Galaxy RNA-seq Tutorials](https://training.galaxyproject.org/training-material/topics/transcriptomics/)
- [nf-core/rnaseq Pipeline](https://nf-co.re/rnaseq) - Community-reviewed workflow
- [RNA-seQC Tool](https://github.com/getzlab/rnaseqc) - Quality metrics for RNA-seq
"""
        )
    ],
    references=[
        "Mantere T, et al. (2022). Long-read sequencing in human genetics. Nature Reviews Genetics, 23(11), 647-659.",
        "Stark R, et al. (2023). Systematic assessment of transcriptomic preprocessing and RNA-seq analysis workflows reveals performance and biological insights. BMC Bioinformatics, 24(1), 316.",
        "Alser M, et al. (2022). Technology dictates algorithms: Recent developments in read alignment. Genome Biology, 23(1), 1-33.",
        "Gu W, et al. (2022). Data processing and analysis considerations for structural variant detection and interpretation. Genome Medicine, 14(1), 97.",
        "Ababou A, et al. (2023). High-throughput sequencing: Principles, applications, and emerging challenges. Frontiers in Bioinformatics, 3, 1195407."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "What are the best practices for variant calling in whole genome sequencing data?"
    
    # Generate prompt
    prompt = ngs_sequencing_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../ngs_sequencing_prompt.json", "w") as f:
        f.write(ngs_sequencing_prompt.to_json())

    # Load prompt template from JSON
    with open("../ngs_sequencing_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt