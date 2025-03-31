"""Module containing the artificial intelligence research_area prompt template."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create an artificial intelligence prompt template
artificial_intelligence_prompt = BioinformaticsPrompt(
    research_area="Artificial Intelligence in Bioinformatics",
    description=(
        "Artificial intelligence (AI) is transforming bioinformatics by providing powerful computational methods for analyzing complex "
        "biological data. Machine learning and deep learning approaches enable researchers to extract patterns, make predictions, and "
        "gain insights from diverse biological datasets, including genomic sequences, protein structures, and clinical information. "
        "The integration of AI with bioinformatics is accelerating discovery in areas such as drug development, disease diagnosis, "
        "and personalized medicine, opening new avenues for understanding biological systems."
    ),
    key_concepts=[
        "Supervised learning (classification, regression)",
        "Unsupervised learning (clustering, dimensionality reduction)",
        "Deep learning (CNNs, RNNs, transformers)",
        "Feature selection and engineering for biological data",
        "Transfer learning and pre-trained biological models",
        "Interpretable AI and explainable models",
        "Representation learning for biological sequences",
        "Multi-modal data integration",
        "Active learning and uncertainty quantification",
        "AI model evaluation in biological contexts"
    ],
    common_tools=[
        "TensorFlow/Keras/PyTorch - Deep learning frameworks",
        "scikit-learn - Machine learning library",
        "DeepChem - Deep learning for drug discovery",
        "BioPython/Biotite - Biological data processing",
        "ESM/ProtTrans - Protein language models",
        "AlphaFold/RoseTTAFold - Protein structure prediction",
        "DeepVariant - Variant calling with deep learning",
        "scVI/SCANPY - Single-cell data analysis",
        "SHAP/LIME - Model interpretation tools",
        "MLflow/Weights & Biases - Experiment tracking"
    ],
    common_file_formats=[
        {"name": "HDF5/TFRecord", "description": "Efficient storage for large datasets and models"},
        {"name": "ONNX", "description": "Open Neural Network Exchange format for model interoperability"},
        {"name": "NumPy arrays (.npy, .npz)", "description": "Efficient numerical data storage"},
        {"name": "Pickle (.pkl)", "description": "Python object serialization"},
        {"name": "CSV/TSV", "description": "Tabular data for features and annotations"},
        {"name": "FASTA/FASTQ", "description": "Sequence data as input to AI models"},
        {"name": "PDB/mmCIF", "description": "Protein structure data for structural AI applications"}
    ],
    examples=[
        FewShotExample(
            query="How do I develop a machine learning model for predicting protein-ligand binding affinity?",
            context="The user has a dataset of protein-ligand complexes with experimentally determined binding affinities and wants to develop a predictive model.",
            response="""
# Machine Learning for Protein-Ligand Binding Affinity Prediction

## Approach Overview
Building an effective protein-ligand binding affinity prediction model involves:
1. Preparing and preprocessing appropriate data
2. Designing an effective feature representation
3. Selecting and implementing appropriate model architectures
4. Training with proper validation approaches
5. Evaluating and interpreting your model

## Key Decision Points

### Data Preparation and Preprocessing
- **Dataset selection**:
  - PDBbind (General/Refined/Core sets) - most commonly used benchmark
  - BindingDB - larger but more heterogeneous
  - ChEMBL - focus on specific target families
  
- **Data splitting considerations**:
  - Random splitting may overestimate performance due to similar proteins in train/test
  - Time-split (newer complexes in test set) for more realistic evaluation
  - Clustering-based splitting by protein sequence or ligand similarity

- **Quality filtering**:
  - Resolution cutoffs for crystal structures (<2.5Å recommended)
  - Binding data type (Ki, Kd, IC50) consistency
  - Handling of missing data and outliers

### Feature Representation
Choose the most appropriate representation for your data:

- **Structure-based approaches**:
  - 3D voxelized grids with atom type channels
  - Distance/adjacency matrices
  - Geometric graph representations
  - Atom-centered symmetry functions

- **Sequence-based approaches**:
  - Protein language models (ESM, ProtTrans) for embeddings
  - One-hot encoding with sliding windows
  - Position-specific scoring matrices

- **Ligand representation**:
  - Molecular fingerprints (ECFP, MACCS)
  - Graph neural networks
  - SMILES-based embeddings
  - 3D conformer-based features

The most successful recent approaches use graph-based representations for both protein and ligand, capturing the interaction interface.

### Model Architecture Selection
Different architectures have different strengths:

- **Convolutional Neural Networks (CNNs)**:
  - Good for grid-based representations
  - Effective at capturing local patterns
  - Examples: AtomNet, 3D-CNN models

- **Graph Neural Networks (GNNs)**:
  - Best for capturing molecular structure
  - Preserve atom connectivity information
  - Examples: GraphDTA, PAGTNet, DeepGS

- **Transformers and attention mechanisms**:
  - Capture long-range dependencies
  - Effective for sequence-based representations
  - Examples: MolTrans, BindFormer

- **Hybrid approaches**:
  - Combine multiple data modalities
  - Often achieve state-of-the-art performance
  - Examples: DeepDTA (CNN+RNN), SIGN (GNN+Transformer)

### Training Strategy
- **Loss function selection**:
  - Mean Squared Error (MSE) for regression
  - Concordance Index (CI) as a ranking metric
  - Custom functions combining regression and ranking

- **Optimization considerations**:
  - Learning rate scheduling (1e-4 to 1e-5 typical)
  - Batch size impacts (16-128 depending on model)
  - Early stopping based on validation performance

- **Regularization approaches**:
  - Dropout (0.1-0.3 typical for binding prediction)
  - Weight decay (L2 regularization)
  - Data augmentation (conformer generation, rotation/translation)

- **Cross-validation strategy**:
  - k-fold CV with protein family awareness
  - Nested CV for hyperparameter tuning
  - Leave-one-cluster-out for protein families

### Model Evaluation and Interpretation
- **Key performance metrics**:
  - Root Mean Square Error (RMSE)
  - Pearson/Spearman correlation coefficients
  - Concordance Index (CI)
  - R² (coefficient of determination)

- **Baseline comparisons**:
  - Classical scoring functions (Vina, Glide)
  - Simpler ML models (Random Forest, SVR)
  - Published benchmarks (careful with dataset alignment)

- **Interpretability approaches**:
  - Attention visualization for important interactions
  - Integrated gradients for feature importance
  - Perturbation analysis of binding site residues
  - SHAP values for per-atom contributions

## Interpretation Considerations

### Common Challenges
- **Dataset biases**:
  - Overrepresentation of certain protein families
  - Activity cliffs in binding data
  - Publication bias towards positive results

- **Generalization issues**:
  - Poor performance on novel protein targets
  - Sensitivity to protein conformational changes
  - Limited performance on low-similarity compounds

- **Computational considerations**:
  - Memory requirements for 3D grid representations
  - Training time for graph-based models
  - Inference speed for virtual screening applications

### Biological Context Integration
For maximum relevance, integrate biological knowledge:
- Incorporate known binding site information when available
- Consider protein flexibility and conformational ensembles
- Account for water-mediated interactions in binding sites
- Include experimental conditions (pH, temperature) if available

### Applications and Limitations
- **Suitable applications**:
  - Virtual screening prioritization
  - Lead optimization guidance
  - Binding mechanism hypotheses

- **Limitations to communicate**:
  - Not a replacement for experimental validation
  - Lower accuracy for novel protein families
  - Potential issues with metalloprotein binding sites
  - Challenges with entropy-driven binding events

## Resources and References
- DeepBindGPU: https://github.com/Sanofi-Public/Bind_DGPU
- GraphDTA: https://github.com/thinng/GraphDTA
- BindingDB: https://www.bindingdb.org
- PDBbind: http://www.pdbbind.org.cn
"""
        ),
        FewShotExample(
            query="What are the best approaches for using machine learning to classify cancer subtypes from gene expression data?",
            context="The user is working with RNA-seq data from cancer patients and wants to develop ML models to classify different cancer subtypes.",
            response="""
# Machine Learning for Cancer Subtype Classification

## Approach Overview
Developing effective cancer subtype classification models involves:
1. Preprocessing gene expression data appropriately
2. Addressing high-dimensionality challenges
3. Selecting appropriate classification algorithms
4. Implementing robust evaluation strategies
5. Interpreting results in a biological context

## Key Decision Points

### Data Preprocessing and Normalization
- **RNA-seq normalization options**:
  - TPM/FPKM/RPKM for within-sample normalization
  - DESeq2/edgeR normalization for differential analysis
  - Quantile normalization for cross-sample comparison
  - log2 transformation to manage dynamic range

- **Batch effect correction**:
  - ComBat or empirical Bayes approaches
  - RUV (Remove Unwanted Variation)
  - PEER factors for technical variation
  - Caution with over-correction removing biological signal

- **Missing value strategies**:
  - Imputation based on k-nearest neighbors
  - Mean/median imputation by gene
  - Matrix completion methods
  - Consider gene/sample filtering thresholds

### Dimensionality Reduction and Feature Selection
- **Feature selection approaches**:
  - Differential expression-based selection
  - Variance-based filtering
  - Recursive feature elimination
  - L1 regularization (Lasso)
  - Domain knowledge-based gene panels

- **Dimensionality reduction techniques**:
  - Principal Component Analysis (PCA)
  - t-SNE for visualization
  - UMAP for preserving global structure
  - Non-negative Matrix Factorization (NMF) for interpretable components
  - Autoencoders for non-linear representations

- **Considerations for cancer data**:
  - Most informative genes may be cancer hallmarks
  - Pathway-level aggregation can improve interpretability
  - Balance between omics-wide approach and targeted biomarkers

### Model Selection and Implementation
- **Traditional ML algorithms**:
  - Random Forest: robust to overfitting, handles high-dimensional data
  - Support Vector Machines: effective for moderate-sized datasets
  - Gradient Boosting: high performance with proper tuning
  - k-Nearest Neighbors: simple baseline, affected by curse of dimensionality

- **Deep learning considerations**:
  - Multi-layer perceptrons for moderate datasets
  - Convolutional networks if spatial structure exists
  - Graph neural networks for pathway-aware approaches
  - Transformer-based models for leveraging pre-trained representations

- **Multi-class strategy selection**:
  - One-vs-Rest for clear subtype boundaries
  - Hierarchical classification for nested subtypes
  - Multi-class direct approaches for balanced classes
  - Consider ordinal approaches for progressive subtypes

### Evaluation Strategy
- **Cross-validation approaches**:
  - Stratified k-fold for class imbalance
  - Nested CV for hyperparameter tuning
  - Leave-one-out for small datasets
  - Patient-aware splits to avoid data leakage

- **Metrics selection**:
  - Balanced accuracy for imbalanced subtypes
  - Macro-averaged F1 score across subtypes
  - Confusion matrix for subtype-specific performance
  - Cohen's kappa for agreement assessment

- **Calibration and uncertainty**:
  - Probability calibration for confidence scores
  - Confidence intervals through bootstrap
  - Uncertainty quantification for clinical relevance
  - Out-of-distribution detection for novel subtypes

## Interpretation Considerations

### Biological Interpretation
- **Pathway enrichment analysis**:
  - Gene set enrichment for important features
  - Network analysis of discriminative genes
  - Integration with protein-protein interaction data
  - Comparison with known cancer driver genes

- **Clinical correlation**:
  - Survival analysis by predicted subtype
  - Treatment response patterns
  - Integration with clinical variables
  - Correlation with established biomarkers

- **Multi-omics integration opportunities**:
  - Combine with DNA methylation
  - Add somatic mutation profiles
  - Consider copy number alterations
  - Integrate with proteomics when available

### Common Challenges
- **Class imbalance**:
  - Rare cancer subtypes underrepresented
  - Synthetic approaches (SMOTE) may introduce artifacts
  - Consider hierarchical approaches for rare subtypes
  - Class weighting rather than undersampling

- **Overfitting risks**:
  - High-dimensional data with few samples
  - Cross-validation may overestimate performance
  - Independent validation cohorts critical
  - Regularization importance increases with feature count

- **Interpretability vs. performance trade-off**:
  - Simpler models may be more interpretable but less accurate
  - Feature importance methods for complex models
  - Consider interpretable by design approaches
  - Balance biological plausibility with predictive power

### Translation to Clinical Application
- **Model deployment considerations**:
  - Standardized data processing pipeline
  - Version control for models and data transforms
  - Regular retraining with new data
  - Clinician-friendly interfaces

- **Validation requirements**:
  - Independent cohort validation
  - Prospective clinical testing
  - Regulatory considerations (FDA, etc.)
  - Integration with existing diagnostic workflows

- **Limitations to communicate**:
  - Novel subtypes may not be well-characterized
  - Batch effects between training and new data
  - Technical platform differences
  - Population differences from training cohorts

## Resources and References
- scikit-learn tutorials: https://scikit-learn.org/stable/tutorial/
- TCGA Pan-Cancer Atlas: https://www.cell.com/pb-assets/consortium/pancanceratlas/pancani3/index.html
- cBioPortal: https://www.cbioportal.org/
- SCANPY tutorials: https://scanpy.readthedocs.io/en/stable/tutorials.html
"""
        ),
        FewShotExample(
            query="How can I use deep learning to predict protein structure from sequence?",
            context="The user wants to understand modern deep learning approaches for protein structure prediction following the success of AlphaFold2.",
            response="""
# Deep Learning for Protein Structure Prediction

## Approach Overview
Modern protein structure prediction with deep learning involves:
1. Understanding the evolution of structure prediction methods
2. Selecting appropriate model architectures
3. Preparing sequence and evolutionary data
4. Implementing training strategies
5. Evaluating and interpreting structural predictions

## Key Decision Points

### Method Selection
Consider these approaches based on your specific needs:

- **End-to-end deep learning models**:
  - AlphaFold2-like architectures (state-of-the-art accuracy)
  - RoseTTAFold (similar approach, more accessible)
  - ESMFold (language model-based, faster but less accurate)
  
- **Template-based approaches**:
  - Templates + deep learning refinement
  - I-TASSER with neural network components
  - ModRefiner for structure refinement
  
- **Fragment assembly with deep learning**:
  - Hybrid methods combining fragment libraries with neural networks
  - DeepFragLib for improved fragment selection
  - End-to-end differentiable assembly

- **Special-case models**:
  - Membrane protein-specific models
  - Models for intrinsically disordered regions
  - Complex assembly predictors (multimer prediction)

### Input Preparation and Feature Engineering
- **Multiple sequence alignment (MSA) generation**:
  - Database selection (UniRef, BFD, MGnify)
  - Search iterations and e-value thresholds
  - MSA depth vs. computation tradeoffs
  - Filtering strategies for diverse MSAs

- **Template selection strategies**:
  - PDB search parameters (coverage, sequence identity)
  - Multiple template combination
  - Obsolete structure handling
  - Quality-based template weighting

- **Input feature representation**:
  - MSA embedding approaches
  - Paired and extra MSA handling
  - Template features (distances, angles, orientations)
  - Residue-wise and pair-wise features

### Model Architecture Considerations
- **Attention mechanism applications**:
  - MSA Transformer blocks (row and column attention)
  - Template attention modules
  - Iterative refinement with attention
  - Evoformer-style architectures

- **Geometric constraints incorporation**:
  - Distance and angle prediction heads
  - End-to-end differentiable structure modules
  - Structure-based loss functions
  - Equivariant neural networks

- **Multi-task learning opportunities**:
  - Secondary structure prediction
  - Solvent accessibility
  - Contact prediction
  - Disorder prediction

- **Inference optimization**:
  - Recycling and iterative refinement
  - Ensembling strategies
  - Temperature sampling for diverse predictions
  - Model size vs. accuracy tradeoffs

### Evaluation and Validation
- **Structure quality assessment metrics**:
  - TM-score for global topology
  - RMSD for specific regions
  - GDT-TS for partial alignment quality
  - lDDT-Cα for local distance agreement
  - pLDDT for per-residue confidence

- **Model confidence interpretation**:
  - AlphaFold2 pLDDT score ranges
  - PAE (predicted aligned error) matrices
  - Ensembling for uncertainty estimation
  - B-factor prediction correlation

- **Functional implication assessment**:
  - Active site geometry inspection
  - Ligand binding site evaluation
  - Protein-protein interaction interfaces
  - Transmembrane region orientation

## Interpretation Considerations

### Structural Biology Integration
- **Experimental validation opportunities**:
  - Targeted regions for crystallography
  - Cryo-EM fitting with predicted models
  - Crosslinking mass spectrometry
  - SAXS envelope alignment

- **Functional hypothesis generation**:
  - Active site residue identification
  - Mutation effect prediction
  - Conformational dynamics inference
  - Allosteric site detection

- **Protein engineering applications**:
  - Stability enhancement
  - Interface design
  - De novo protein design
  - Enzyme activity optimization

### Limitations and Challenges
- **Common prediction limitations**:
  - Highly flexible or disordered regions
  - Proteins requiring cofactors or partners
  - Post-translational modifications
  - Alternate conformational states

- **Technical challenges**:
  - Computational resource requirements
  - MSA generation for orphan sequences
  - Very large protein structures (>1500 residues)
  - Handling of non-standard residues

- **Biological context considerations**:
  - Cellular environment effects
  - Conformational ensembles vs. static structures
  - Functional relevance of predicted states
  - Species-specific folding differences

### Emerging Directions
- **End-to-end protein design**:
  - Inverse folding (sequence from structure)
  - Jointly trained design-fold networks
  - Function-guided sequence optimization
  - Multi-state design for dynamics

- **Complex structure prediction**:
  - Heteromeric complexes beyond dimers
  - Protein-ligand complex prediction
  - Nucleic acid-protein interactions
  - Membrane protein-lipid interactions

- **Integration with other omics data**:
  - Expression-aware structure prediction
  - Variation impact on structure
  - Systems-level structural proteomics
  - Species-specific structural models

## Resources and References
- ColabFold: https://github.com/sokrypton/ColabFold
- AlphaFold2 GitHub: https://github.com/deepmind/alphafold
- RoseTTAFold: https://github.com/RosettaCommons/RoseTTAFold
- ESMFold: https://github.com/facebookresearch/esm
"""
        )
    ],
    references=[
        "Jumper J, et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature.",
        "Vaswani A, et al. (2017). Attention is all you need. Advances in Neural Information Processing Systems.",
        "Eraslan G, et al. (2019). Deep learning: new computational modelling techniques for genomics. Nature Reviews Genetics.",
        "Ching T, et al. (2018). Opportunities and obstacles for deep learning in biology and medicine. Journal of The Royal Society Interface.",
        "Zou J, et al. (2019). A primer on deep learning in genomics. Nature Genetics.",
        "Senior AW, et al. (2020). Improved protein structure prediction using potentials from deep learning. Nature."
    ]
)

# Export the prompt for use in the package
if __name__ == "__main__":
    # Test the prompt with a sample query
    user_query = "How can I use deep learning to analyze single-cell RNA-seq data?"
    
    # Generate prompt
    prompt = artificial_intelligence_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../artificial_intelligence_prompt.json", "w") as f:
        f.write(artificial_intelligence_prompt.to_json())

   # Load prompt template from JSON
    with open("../artificial_intelligence_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt