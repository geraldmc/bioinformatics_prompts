import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a workflow automation prompt template
workflow_automation_prompt = BioinformaticsPrompt(
    research_area="Bioinformatics Workflow Automation",
    description=(
        "Workflow automation in bioinformatics streamlines complex data analysis pipelines, "
        "reducing manual intervention, minimizing errors, and enhancing reproducibility. "
        "Modern workflow systems allow researchers to focus on experimental design and "
        "result interpretation rather than computational implementation details. "
        "These systems orchestrate complex analyses across computing environments, "
        "track data provenance, and facilitate collaboration through well-defined, "
        "shareable pipeline definitions."
    ),
    key_concepts=[
        "Workflow definition languages (WDL, CWL, Nextflow DSL)",
        "Container technologies (Docker, Singularity) for environment reproducibility",
        "Pipeline modularity and reusability",
        "Data provenance tracking",
        "Batch processing and parallelization",
        "Cloud and HPC integration",
        "Dependency management",
        "Version control for workflows"
    ],
    common_tools=[
        "Nextflow - Stream-based workflow language and execution platform",
        "Snakemake - Python-based workflow management system",
        "WDL/Cromwell - Workflow Definition Language with Cromwell execution engine",
        "Galaxy - Web-based workflow platform for accessible bioinformatics",
        "CWL - Common Workflow Language specification",
        "Airflow - Platform for programmatically authoring, scheduling and monitoring workflows",
        "Parsl - Parallel Scripting Library for Python-based workflows"
    ],
    common_file_formats=[
        {"name": "YAML/JSON", "description": "For workflow configuration and parameter specifications"},
        {"name": ".nf/.config", "description": "Nextflow workflow and configuration files"},
        {"name": "Snakefile", "description": "Snakemake workflow definition files"},
        {"name": ".wdl", "description": "Workflow Definition Language files"},
        {"name": ".cwl", "description": "Common Workflow Language files"},
        {"name": "Dockerfile/Singularity", "description": "Container definition files for reproducible environments"},
        {"name": "workflow.ga", "description": "Galaxy workflow export format"}
    ],
    examples=[
        FewShotExample(
            query="How should I design a Nextflow pipeline for RNA-seq analysis that can scale to hundreds of samples?",
            context="The user needs to process RNA-seq data for a large-scale project and wants to ensure the pipeline is scalable, reproducible, and robust.",
            response="""
# RNA-seq Analysis Workflow Design with Nextflow

## Conceptual Approach

For a scalable RNA-seq pipeline handling hundreds of samples, you should focus on these key design principles:

1. **Modular process design**: Break your workflow into independent, reusable processes
2. **Containerization**: Use Docker/Singularity containers for each process
3. **Configuration separation**: Separate your pipeline logic from execution parameters
4. **Resource management**: Implement dynamic resource allocation
5. **Built-in quality control**: Include QC checks between major stages

## Decision Framework

### Workflow Structure Decisions
- **Input handling**: Design a flexible input system that can handle different metadata formats
- **Process granularity**: Determine the appropriate level of process separation
- **Checkpointing strategy**: Strategic placement of publishDir directives for intermediate results
- **Parameter scoping**: Which parameters should be process-specific vs. global
- **Execution profile design**: Development vs. production environments

### Process-Specific Decisions
- **Quality control**: FastQC pre-trim and post-trim with MultiQC integration
- **Read trimming**: Software selection (Trimmomatic, fastp) with parameter considerations
- **Alignment choice**: STAR vs. Salmon vs. hybrid approach
- **Quantification strategy**: Gene-level vs. transcript-level quantification
- **Batch processing strategy**: Sample-level parallelism vs. batch chunking

### Resource Considerations
- **Memory strategy**: How to dynamically allocate memory based on input size
- **Storage management**: Temporary vs. persistent storage decisions
- **Execution environment**: Local, cluster, or cloud considerations
- **Container strategy**: Single container vs. process-specific containers

## Implementation Outline

Create a project structure like:
```
my_rnaseq_pipeline/
├── main.nf              # Main workflow definition
├── nextflow.config      # Pipeline configuration
├── conf/                # Environment-specific configs
├── modules/             # Process definitions
├── containers/          # Dockerfiles for processes
└── bin/                 # Helper scripts
```

Focus on designing your main.nf as a high-level orchestration of modules:

```groovy
// High-level workflow structure
workflow {
    reads_ch = channel.fromFilePairs(params.reads)
    metadata_ch = channel.fromPath(params.metadata)
    
    FASTQC(reads_ch)
    TRIMMING(reads_ch)
    FASTQC_POST(TRIMMING.out)
    ALIGNMENT(TRIMMING.out)
    QUANTIFICATION(ALIGNMENT.out, metadata_ch)
    MULTIQC(
        FASTQC.out,
        FASTQC_POST.out,
        ALIGNMENT.out.stats,
        QUANTIFICATION.out.stats
    )
}
```

## Key Considerations

- **Error handling**: Build robust error detection and reporting
- **Resume capabilities**: Structure for efficient restart after failures
- **Monitoring**: Integrate workflow monitoring for long-running jobs
- **Data provenance**: Ensure all pipeline parameters are captured in output
- **Validation**: Implement input validation to prevent downstream failures

## Useful Resources

- [Nextflow Patterns](https://nextflow-io.github.io/patterns/) - Common design patterns
- [nf-core](https://nf-co.re/) - Collection of peer-reviewed Nextflow pipelines
- [Nextflow Tower](https://tower.nf/) - Monitoring and execution platform
- [Seqera Labs Tutorials](https://seqera.io/training/) - Advanced Nextflow training materials
"""
        ),
        FewShotExample(
            query="What are the key design decisions when choosing between Snakemake and Nextflow for a metagenomics analysis pipeline?",
            context="The user is designing a new metagenomics pipeline and needs to understand the tradeoffs between different workflow management systems.",
            response="""
# Workflow Manager Selection for Metagenomics Analysis

## Conceptual Framework

When choosing between Snakemake and Nextflow for metagenomics analysis, consider these key dimensions:

1. **Team expertise and familiarity**
2. **Pipeline complexity and data flow patterns**
3. **Execution environment requirements**
4. **Reproducibility needs**
5. **Community and ecosystem support**

## Decision Framework

### Language and Syntax Considerations

#### Snakemake
- **Python-based**: Leverages Python for scripting and rule definition
- **Declarative approach**: Focuses on defining output files and their dependencies
- **Learning curve**: More intuitive for researchers with Python experience
- **Integration**: Seamless integration with other Python scientific libraries

#### Nextflow
- **Groovy-based DSL**: Uses Groovy for its domain-specific language
- **Dataflow paradigm**: Built around asynchronous channels and processes
- **Expressiveness**: More powerful for complex data transformations
- **Learning curve**: Steeper for teams without JVM language experience

### Execution Model Decisions

#### Snakemake
- **Job scheduling**: Rule-based execution with implicit dependency resolution
- **Parallelization**: Automatic parallelization based on resource availability
- **Cluster integration**: Supports most HPC schedulers with generic interface
- **Cloud support**: AWS, Google Cloud, and other cloud support via generic interfaces

#### Nextflow
- **Streaming execution**: Process-based dataflow model for streaming execution
- **Executors**: Native support for Kubernetes, AWS Batch, Google Cloud, etc.
- **Containerization**: Deeper integration with Docker/Singularity
- **Resume capability**: More granular resume functionality for failed runs

### Metagenomics-Specific Considerations
- **Data heterogeneity**: How well does the workflow handle diverse input types?
- **Process variation**: Can the workflow adapt to different analysis strategies?
- **Resource management**: How efficiently are compute resources allocated?
- **Pipeline modularity**: Support for conditional workflows and dynamic rules
- **Assembly approach**: Assembly-first vs. read-based analysis implementation

## Implementation Considerations

### Snakemake Approach
```python
# High-level workflow structure
rule all:
    input:
        "results/final_report.html"

rule quality_control:
    input:
        "data/samples/{sample}_R{read}.fastq.gz"
    output:
        "results/qc/{sample}_R{read}_filtered.fastq.gz"
    
rule assembly:
    input:
        r1="results/qc/{sample}_R1_filtered.fastq.gz",
        r2="results/qc/{sample}_R2_filtered.fastq.gz"
    output:
        "results/assembly/{sample}/contigs.fa"
```

### Nextflow Approach
```groovy
// High-level workflow structure
workflow {
    reads_ch = Channel.fromFilePairs("data/samples/*_R{1,2}.fastq.gz")
    
    QUALITY_CONTROL(reads_ch)
    ASSEMBLY(QUALITY_CONTROL.out)
    BINNING(ASSEMBLY.out.contigs, QUALITY_CONTROL.out)
    ANNOTATION(BINNING.out.bins)
}
```

## Key Differentiators

| Feature | Snakemake | Nextflow |
|---------|-----------|----------|
| **Community** | Strong academic community | Growing industry adoption |
| **File focus** | File-based dependencies | Channel-based dataflows |
| **Scaling** | Good for batch processing | Better for streaming data |
| **Visualization** | DAG visualization built-in | Requires Tower or third-party tools |
| **Containers** | Container support | Native container integration |

## Resources

- [Snakemake Documentation](https://snakemake.readthedocs.io/)
- [Nextflow Documentation](https://www.nextflow.io/docs/latest/index.html)
- [Workflow Comparison Paper](https://doi.org/10.1093/bioinformatics/btaa742)
- [nf-core Metagenomic Pipelines](https://nf-co.re/pipelines?q=metagenomics)
"""
        )
    ],
    references=[
        "Leipzig J. (2017). A review of bioinformatic pipeline frameworks. Briefings in Bioinformatics, 18(3), 530-536.",
        "Ewels PA, et al. (2020). The nf-core framework for community-curated bioinformatics pipelines. Nature Biotechnology, 38(3), 276-278.",
        "Koster J, Rahmann S. (2012). Snakemake--a scalable bioinformatics workflow engine. Bioinformatics, 28(19), 2520-2522.",
        "Di Tommaso P, et al. (2017). Nextflow enables reproducible computational workflows. Nature Biotechnology, 35(4), 316-319."
    ]
)

# Save the workflow automation prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "How do I design a scalable and reproducible bioinformatics workflow for bacterial genome assembly and annotation?"
    
    # Generate prompt
    prompt = workflow_automation_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../workflow_automation_prompt.json", "w") as f:
        f.write(workflow_automation_prompt.to_json())

   # Load prompt template from JSON
    with open("../workflow_automation_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt