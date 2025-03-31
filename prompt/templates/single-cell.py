import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a single-cell genomics prompt template
single_cell_genomics_prompt = BioinformaticsPrompt(
    research_area="Single-Cell Genomics",
    description=(
        "Single-cell genomics examines the genetic and transcriptomic profiles of individual cells, "
        "revealing cellular heterogeneity masked in bulk analyses. These techniques enable researchers "
        "to identify rare cell populations, trace developmental trajectories, and understand cell-to-cell "
        "variability in complex tissues and disease states. By profiling thousands to millions of cells "
        "simultaneously, single-cell approaches provide unprecedented resolution into cellular diversity, "
        "lineage relationships, and dynamic biological processes."
    ),
    key_concepts=[
        "Cellular heterogeneity and rare cell type identification",
        "Dimensionality reduction and visualization (PCA, t-SNE, UMAP)",
        "Cell clustering and annotation",
        "Trajectory inference and pseudotime analysis",
        "Differential expression analysis at single-cell level",
        "Batch effect correction and data integration",
        "Gene regulatory network inference",
        "Multi-modal single-cell analysis (RNA, ATAC, protein)"
    ],
    common_tools=[
        "Seurat - R package for single-cell analysis",
        "Scanpy - Python-based single-cell analysis suite",
        "Monocle/Monocle3 - Trajectory analysis",
        "Cell Ranger - 10x Genomics preprocessing pipeline",
        "SCENIC - Gene regulatory network inference",
        "Harmony/Scanorama/LIGER - Batch correction and integration",
        "Velocyto/scVelo - RNA velocity analysis",
        "CellPhoneDB - Cell-cell communication inference"
    ],
    common_file_formats=[
        {"name": "H5AD", "description": "AnnData format for Scanpy (HDF5-based)"},
        {"name": "RDS", "description": "R data serialization for Seurat objects"},
        {"name": "MTX/HDF5", "description": "Sparse matrix formats for count data"},
        {"name": "FASTQ", "description": "Raw sequencing reads"},
        {"name": "BAM", "description": "Aligned sequence data"},
        {"name": "Loom", "description": "Columnar format for large omics datasets"},
        {"name": "CellBrowser", "description": "Web visualization format for single-cell data"}
    ],
    examples=[
        FewShotExample(
            query="What are the key analytical decisions for a scRNA-seq experiment focused on tumor heterogeneity?",
            context="The user is planning a single-cell RNA-seq study of tumor samples and needs guidance on the analytical workflow and decision points.",
            response="""
# Single-Cell RNA-seq Analysis for Tumor Heterogeneity

## Conceptual Framework

Analyzing tumor heterogeneity with scRNA-seq requires a structured decision framework across these key analytical stages:

1. **Pre-processing & quality assessment**
2. **Computational cell type identification**
3. **Tumor microenvironment characterization**
4. **Clonal diversity & evolution analysis**
5. **Functional & pathway interpretation**

## Critical Decision Points

### Experimental Design Decisions
- **Sample preparation**: Fresh vs. frozen tissue considerations
- **Platform selection**: Droplet-based (10x) vs. plate-based (Smart-seq2) tradeoffs
  - Droplet: Higher cell counts but shallower sequencing
  - Plate-based: Deeper sequencing but fewer cells
- **Cell enrichment strategy**: Whole tumor vs. sorted populations 
- **Controls**: Need for matched normal tissue or peripheral blood
- **Multi-modal considerations**: RNA-only vs. RNA+protein/ATAC

### Pre-processing Decisions
- **Quality control thresholds**: Balance between stringency and cell recovery
  - Minimum features (genes): 200-1000 depending on platform
  - Maximum mitochondrial percentage: 10-20% (tumor-dependent)
  - Doublet detection approach: Cell filtration vs. computational removal
- **Normalization strategy**: Global scaling vs. SCTransform vs. pool-based
- **Feature selection**: Highly variable gene identification approach
- **Batch correction**: Need and method selection based on experimental design

### Clustering & Cell Type Identification
- **Dimensionality reduction**: PCA components to retain (30-50 typical)
- **Clustering resolution**: Low (major cell types) vs. high (subtle states)
- **Visualization approach**: t-SNE vs. UMAP parameters
- **Cell type annotation**: Reference-based vs. marker-driven approaches
  - References: Established atlases vs. custom references
  - Markers: Literature-derived vs. computationally identified

### Tumor-Specific Analysis
- **Malignant cell identification**: CNV inference vs. marker-based approaches
- **Stromal/immune profiling**: General markers vs. specialized references
- **Cancer subtype classification**: De novo clustering vs. signature scoring
- **Pathway activity estimation**: GSEA vs. AUCell vs. Metascape
- **Intratumoral heterogeneity quantification**: Continuous vs. discrete approaches

### Advanced Analyses
- **Developmental trajectory inference**: Method selection based on biological context
  - Linear processes: Slingshot, Monocle2
  - Tree-like processes: Monocle3, PAGA
- **RNA velocity**: Splicing-based (Velocyto) vs. dynamical model (scVelo)
- **Cell-cell interaction**: Tools like CellPhoneDB, CellChat, or NicheNet
- **Gene regulatory network**: SCENIC vs. integrated approaches

## Implementation Framework

Your analytical approach should prioritize these considerations:

- **Alignment with biological question**: Focus on markers and methods specific to cancer biology
- **Robustness assessment**: Confirm findings persist across clustering parameters
- **Orthogonal validation**: Plan for spatial or functional validation of key findings
- **Integration capability**: Position your analysis for integration with other tumors/datasets

## Useful Resources

- [Tumor Cell Atlas Consortium](https://www.humancellatlas.org/biological-networks/)
- [scRNA-seq analysis tutorial with Seurat](https://satijalab.org/seurat/articles/pbmc3k_tutorial.html)
- [Scanpy tutorials for cancer analysis](https://scanpy-tutorials.readthedocs.io/)
- [Best practices for single-cell analysis](https://doi.org/10.1038/s41576-020-0276-2)
"""
        ),
        FewShotExample(
            query="How do I integrate multiple single-cell datasets from different batches and technologies?",
            context="The user has several scRNA-seq datasets generated with different platforms (10x, Smart-seq2) and needs to integrate them for a unified analysis.",
            response="""
# Integration of Heterogeneous Single-Cell Datasets

## Conceptual Framework

Integrating single-cell datasets across batches and technologies requires addressing three core challenges:

1. **Technical variation**: Platform-specific biases and batch effects
2. **Biological variation**: True differences in cell states and types
3. **Data structure compatibility**: Reconciling different feature spaces

The integration process involves these key stages:
- Pre-integration quality control
- Feature space harmonization
- Batch effect correction
- Integrated analysis
- Post-integration validation

## Decision Framework

### Pre-Integration Assessment
- **Dataset compatibility evaluation**:
  - Cell type overlap: Substantial vs. minimal shared populations
  - Sequencing depth differences: Shallow vs. deep profiling
  - Protocol differences: UMI-based vs. full-length
  - Experimental conditions: Time points, treatments, tissues
- **Reference selection**: Use highest-quality dataset or external reference?
- **Analysis scope**: Full transcriptome vs. conserved highly variable features

### Feature Space Decisions
- **Gene set selection strategy**:
  - Highly variable genes in each dataset independently
  - Genes variable across all datasets
  - Curated marker panels for expected cell types
  - Platform-agnostic features (protein-coding genes only)
- **Feature space dimensions**: Number of genes to include (1000-5000 typical)
- **Anchor features**: Cell type markers vs. housekeeping genes as guides

### Integration Method Selection
- **Method choice based on dataset relationships**:
  - **Canonical Correlation Analysis (Seurat)**: Best for partially overlapping populations
  - **Mutual Nearest Neighbors (Scanorama)**: Robust for varied cell type compositions
  - **Harmony**: Effective for multiple batches with shared cell types
  - **LIGER**: Useful for finding shared and dataset-specific patterns
  - **scVI**: Deep learning approach handling complex batch effects
  - **BBKNN**: Fast for very large datasets

### Integration Parameters
- **Anchor selection**: Number and quality threshold of anchors/mutual neighbors
- **Dimensionality**: Number of components/dimensions for integration
- **Regularization**: Strength of batch effect correction vs. preservation of biological signal
- **Scaling factors**: Accounting for different sequencing depths

### Post-Integration Analysis
- **Integration quality assessment**:
  - Batch mixing metrics (kBET, LISI scores)
  - Conservation of known biological signals
  - Cell type clustering by biological rather than technical factors
- **Label transfer approach**: Confidence thresholds for annotation propagation
- **Differential expression strategy**:
  - Paired approach with batch as covariate
  - Dataset-specific vs. integrated testing
  - Pseudobulk vs. single-cell level testing

## Implementation Considerations

Your integration strategy should prioritize:
- **Biological signal preservation**: Test with known markers before/after integration
- **Conservative correction**: Start with mild integration parameters, then increase if needed
- **Incremental integration**: For many datasets, consider hierarchical integration strategies
- **Validation**: Confirm integrated results with orthogonal methods or known biology

## Useful Resources

- [Harmony integration tutorial](https://github.com/immunogenomics/harmony)
- [Seurat integration vignette](https://satijalab.org/seurat/articles/integration_introduction.html)
- [Scanpy integration examples](https://scanpy.readthedocs.io/en/stable/tutorials.html)
- [Benchmarking of integration methods](https://doi.org/10.1038/s41592-021-01336-8)
"""
        )
    ],
    references=[
        "Luecken MD, Theis FJ. (2019). Current best practices in single‐cell RNA‐seq analysis: a tutorial. Molecular Systems Biology, 15(6), e8746.",
        "Stuart T, Satija R. (2019). Integrative single-cell analysis. Nature Reviews Genetics, 20(5), 257-272.",
        "Lähnemann D, et al. (2020). Eleven grand challenges in single-cell data science. Genome Biology, 21(1), 1-35.",
        "Chen G, et al. (2019). Single-cell multi-omics technology: methodology and application. Frontiers in Cell and Developmental Biology, 7, 260."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "How do I interpret cell clusters in my single-cell RNA-seq data from a mixed tissue sample?"
    
    # Generate prompt
    prompt = single_cell_genomics_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../single_cell_genomics_prompt.json", "w") as f:
        f.write(single_cell_genomics_prompt.to_json())

   # Load prompt template from JSON
    with open("../single_cell_genomics_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt