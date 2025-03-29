import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a bioinformatics tools prompt template
bioinformatics_tools_prompt = BioinformaticsPrompt(
    discipline="Bioinformatics Tool Selection and Evaluation",
    description=(
        "Bioinformatics tools encompass software, algorithms, and databases that facilitate biological data analysis "
        "across diverse research areas. The selection of appropriate tools is a critical decision in bioinformatics "
        "workflows, requiring careful evaluation of tool capabilities, limitations, computational requirements, and "
        "suitability for specific research questions. This evaluation process involves understanding the principles, "
        "algorithms, input/output formats, and validation approaches to ensure reliable and reproducible results."
    ),
    key_concepts=[
        "Tool evaluation criteria (accuracy, speed, resource requirements, maintenance)",
        "Usability spectrum (command-line, web interfaces, programmable APIs)",
        "Validation approaches and benchmarking strategies",
        "Containerization and environment management",
        "Interoperability and data format compatibility",
        "Version control and reproducibility considerations",
        "Resource scaling and high-performance computing integration",
        "Tool documentation and community support"
    ],
    common_tools=[
        "Bioconda - Package management system for bioinformatics",
        "Galaxy - Web-based platform for accessible bioinformatics",
        "Docker/Singularity - Container platforms for tool deployment",
        "Conda/Mamba - Environment management systems",
        "Bioconductor - R packages for bioinformatics",
        "Biopython/Bioperl - Programming toolkits for biological computation",
        "CyVerse/Terra - Cloud platforms for bioinformatics analysis"
    ],
    common_file_formats=[
        {"name": "YAML/JSON", "description": "Tool configuration and metadata formats"},
        {"name": "Dockerfile/Singularity", "description": "Container definition files"},
        {"name": "environment.yml", "description": "Conda environment specifications"},
        {"name": "requirements.txt", "description": "Python dependency specifications"},
        {"name": "CWL/WDL/Nextflow", "description": "Workflow definition files referencing tools"},
        {"name": "XML", "description": "Tool definitions in Galaxy and other systems"}
    ],
    examples=[
        FewShotExample(
            query="How do I evaluate and select the most appropriate variant calling tools for my whole genome sequencing project?",
            context="The user is planning a human whole genome sequencing study focused on rare disease diagnosis and needs to choose variant calling tools.",
            response="""
# Evaluating Variant Calling Tools for Whole Genome Sequencing

## Conceptual Framework

Selecting variant callers for WGS analysis requires evaluating tools across multiple dimensions:

1. **Variant type coverage**: SNVs, indels, structural variants, copy number
2. **Performance characteristics**: Sensitivity, specificity, computational demands
3. **Use case alignment**: Research, clinical, or population applications
4. **Integration potential**: With existing pipelines and downstream tools

## Decision Framework

### Project Requirement Assessment
- **Study objectives**: Research discovery vs. clinical diagnosis vs. population screening
- **Variant classes of interest**: 
  - Simple variants (SNVs/indels)
  - Structural variants (deletions, duplications, inversions)
  - Copy number variants
  - Mobile element insertions
- **Sequencing characteristics**:
  - Coverage depth (30x, 60x, etc.)
  - Read length and sequencing technology (short vs. long reads)
  - Library preparation method
- **Computational constraints**:
  - Available compute resources
  - Time constraints
  - Scalability requirements (single vs. many samples)

### Tool Category Selection
- **Comprehensive callers**:
  - GATK HaplotypeCaller: Gold standard with high specificity
  - DeepVariant: Neural network-based with high accuracy
  - Strelka2: Fast and accurate for germline variants
- **Structural variant callers**:
  - Manta: Balanced speed and sensitivity
  - DELLY: High sensitivity for deletions
  - GRIDSS: Assembly-based for complex events
- **Copy number variant callers**:
  - CNVnator: Read-depth based detection
  - Canvas: Combines multiple signals
  - FALCON: Long-read phased SV detection
- **Multi-algorithm approaches**:
  - Consensus calling (GATK+Strelka2+DeepVariant)
  - Ensemble methods (Parliament2, SV-Callers)

### Evaluation Criteria
- **Accuracy metrics**:
  - Sensitivity/recall: Proportion of true variants detected
  - Precision: Proportion of calls that are true variants
  - F1 score: Harmonic mean of precision and recall
  - False discovery rate: Proportion of false calls
- **Computational metrics**:
  - Runtime on standard datasets
  - Memory requirements
  - Parallelization capabilities
  - Cloud compatibility
- **Practical considerations**:
  - Ease of installation and configuration
  - Quality of documentation
  - Active maintenance and community support
  - Compatibility with downstream tools (VEP, ANNOVAR)

### Benchmarking Strategy
- **Truth sets comparison**:
  - Genome in a Bottle (GIAB) for standard benchmarking
  - Synthetic benchmarks (Bam Surgeon, NEAT)
  - Family-based validation (inheritance pattern checking)
- **Validation approaches**:
  - Orthogonal technologies (arrays, targeted sequencing)
  - Technical replicates consistency
  - Cross-tool concordance analysis

## Implementation Considerations

- **Start with established pipelines**:
  - Consider nf-core/sarek or GATK best practices as foundations
  - Adapt based on specific project needs
- **Variant classification strategies**:
  - Filter based on quality metrics specific to each tool
  - Apply tool-specific recommended filtering thresholds
  - Consider joint filtering across multiple tools for consensus calls
- **Validation strategy**:
  - Confirm subset of variants with orthogonal methods
  - Check inheritance patterns in family studies
  - Compare performance on known benchmarking datasets

## Useful Resources

- [Genome in a Bottle Consortium](https://www.nist.gov/programs-projects/genome-bottle)
- [GA4GH Benchmarking Framework](https://github.com/ga4gh/benchmarking-tools)
- [nf-core/sarek pipeline](https://nf-co.re/sarek)
- [GATK Best Practices](https://gatk.broadinstitute.org/hc/en-us/articles/360035894711)
- [PrecisionFDA Truth Challenges](https://precision.fda.gov/challenges)
"""
        ),
        FewShotExample(
            query="How should I approach tool selection for multi-omics data integration in a cancer research project?",
            context="The user has RNA-seq, whole-genome sequencing, and proteomics data from cancer patients and wants to integrate these datasets.",
            response="""
# Tool Selection for Multi-Omics Cancer Data Integration

## Conceptual Framework

Multi-omics integration requires tools that can:
1. **Handle heterogeneous data types**: Genomic, transcriptomic, proteomic
2. **Address different scales and distributions**: Across various measurement technologies
3. **Implement appropriate integration strategies**: Early, intermediate, or late integration
4. **Provide interpretable outputs**: For biological insight extraction

## Decision Framework

### Integration Strategy Selection
- **Integration level**:
  - **Early integration**: Combine raw or processed data before analysis
    - Higher information retention but higher dimensionality challenges
  - **Intermediate integration**: Transform each dataset then integrate
    - Balances information and dimensionality
  - **Late integration**: Analyze each dataset separately, then combine results
    - Simpler but may miss cross-omics interactions
- **Analysis objective**:
  - Patient stratification and subtyping
  - Biomarker discovery
  - Regulatory network reconstruction
  - Causal mechanism identification

### Tool Category Selection

#### Data Preprocessing Tools
- **Quality control and normalization**:
  - RNAseq: MultiQC, DESeq2 normalization
  - WGS: FastQC, Picard metrics
  - Proteomics: MaxQuant, MSstats
- **Batch effect correction**:
  - ComBat/ComBat-seq: Parametric adjustment
  - Harmony: Fast integration via soft clustering
  - PEER: Factor analysis for hidden confounders

#### Integration Approaches
- **Matrix factorization methods**:
  - iCluster+/iClusterPlus: Joint clustering across omics
  - MOFA/MOFA+: Factor analysis for multi-omics
  - NMF-based methods: Identification of metagenes
- **Network-based methods**:
  - SNF (Similarity Network Fusion): Patient similarity networks
  - mixOmics: Sparse multi-omics integration
  - DIABLO: Discriminant analysis integration
- **Deep learning approaches**:
  - MOMA: Multi-omics autoencoder
  - OmiEmbed: Representation learning for integration
  - VAEs for multi-modal data integration

#### Visualization and Interpretation
- **Multi-omics visualization**:
  - mixOmics: Correlation circle plots
  - Clustergrammer: Interactive heatmaps
  - omicsCIR: Circular visualizations for integration
- **Pathway analysis**:
  - PathwayPCA: Pathway analysis on multi-omics
  - ReactomeGSA: Multi-omics pathway analysis
  - OmicsAnalyst: Visual analytics for integration

### Tool Evaluation Criteria
- **Data type compatibility**:
  - Supported omics data types
  - Required data formats and preprocessing
  - Missing data handling capabilities
- **Statistical approach**:
  - Underlying mathematical framework
  - Assumptions about data distributions
  - Ability to handle high-dimensional, sparse data
- **Computational considerations**:
  - Scalability to large cohorts
  - Memory and runtime requirements
  - GPU acceleration availability
- **Biological interpretability**:
  - Connection to known pathways and mechanisms
  - Visualization capabilities
  - Integration with knowledge bases

### Practical Implementation Decisions
- **Data harmonization approach**:
  - Feature matching across platforms
  - Sample alignment and missing data handling
  - Batch effect identification and correction
- **Feature selection strategy**:
  - Biology-driven vs. data-driven selection
  - Variance-based filtering per omics layer
  - Cross-omics correlation filtering
- **Validation strategy**:
  - Cross-validation schemes appropriate for multi-omics
  - Independent cohort validation
  - Functional validation of findings

## Key Considerations

- **Start simple**: Begin with pairwise integration before full multi-omics
- **Biological knowledge**: Incorporate pathway information to guide integration
- **Iterative approach**: Refine integration as insights emerge
- **Benchmarking**: Compare multiple tools on subsets of data
- **Interpretability**: Prioritize methods that produce biologically actionable results

## Useful Resources

- [Multi-Omics Factor Analysis (MOFA)](https://github.com/bioFAM/MOFA2)
- [mixOmics R package](https://mixomics.org/)
- [Similarity Network Fusion](http://compbio.cs.toronto.edu/SNF/)
- [OmicsCIR](https://omicscir.elixir-luxembourg.org/)
- [OmicsAnalyst](https://www.omicsanalyst.ca/)
- [Multi-Omics Benchmarking](https://doi.org/10.1186/s13059-020-02032-0)
"""
        )
    ],
    references=[
        "Nunes M, et al. (2022). Benchmarking of structural variant calling tools. Bioinformatics, 38(5), 1252-1259.",
        "Cantini L, et al. (2021). Benchmarking joint multi-omics dimensionality reduction approaches for the study of cancer. Nature Communications, 12(1), 124.",
        "Mangul S, et al. (2019). Systematic benchmarking of omics computational tools. Nature Communications, 10(1), 1393.",
        "Ye Y, et al. (2021). A comprehensive survey of omics data resources for human diseases. Briefings in Bioinformatics, 22(6)."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "What considerations should I make when selecting alignment tools for my RNA-seq experiment?"
    
    # Generate prompt
    prompt = bioinformatics_tools_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../bioinformatics_tools_prompt.json", "w") as f:
        f.write(bioinformatics_tools_prompt.to_json())

   # Load prompt template from JSON
    with open("../bioinformatics_tools_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt