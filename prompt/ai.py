"""Module containing the artificial intelligence discipline prompt template."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create an artificial intelligence prompt template
artificial_intelligence_prompt = BioinformaticsPrompt(
    discipline="Artificial Intelligence in Bioinformatics",
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
            query="How do I develop a deep learning model for predicting protein-ligand binding affinity?",
            context="The user has a dataset of protein-ligand complexes with experimentally determined binding affinities and wants to develop a predictive model.",
            response= """
# Developing a Deep Learning Model for Protein-Ligand Binding Affinity Prediction

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

```python
# Example of sequence identity-based splitting
from Bio import pairwise2
import numpy as np
from sklearn.model_selection import train_test_split

def calculate_sequence_identity(seq1, seq2):
    alignment = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    identical = sum(a == b for a, b in zip(alignment.seqA, alignment.seqB))
    return identical / min(len(seq1), len(seq2))

def sequence_based_split(proteins, sequences, test_size=0.2, identity_threshold=0.3):
    """Split ensuring test proteins have <30% sequence identity to training proteins"""
    train_indices = []
    test_indices = []
    
    # Start with random split as initialization
    initial_train, initial_test = train_test_split(range(len(proteins)), test_size=test_size)
    train_indices = list(initial_train)
    
    # For each potential test protein
    for i in initial_test:
        # Check if it has >30% identity to any training protein
        high_identity = False
        for j in train_indices:
            if calculate_sequence_identity(sequences[i], sequences[j]) > identity_threshold:
                high_identity = True
                break
        
        if not high_identity:
            test_indices.append(i)
        else:
            train_indices.append(i)
    
    return train_indices, test_indices
```

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

```python
# Example skeleton of a Graph Neural Network for protein-ligand binding
import torch
import torch_geometric

class ProteinLigandGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim=64):
        super(ProteinLigandGNN, self).__init__()
        
        # Graph convolution layers for protein
        self.protein_conv1 = torch_geometric.nn.GCNConv(node_features, hidden_dim)
        self.protein_conv2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)
        
        # Graph convolution layers for ligand
        self.ligand_conv1 = torch_geometric.nn.GCNConv(node_features, hidden_dim)
        self.ligand_conv2 = torch_geometric.nn.GCNConv(hidden_dim, hidden_dim)
        
        # Interaction module
        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, protein_data, ligand_data):
        # Process protein graph
        protein_x = self.protein_conv1(protein_data.x, protein_data.edge_index)
        protein_x = torch.nn.functional.relu(protein_x)
        protein_x = self.protein_conv2(protein_x, protein_data.edge_index)
        protein_x = torch.nn.functional.relu(protein_x)
        
        # Process ligand graph
        ligand_x = self.ligand_conv1(ligand_data.x, ligand_data.edge_index)
        ligand_x = torch.nn.functional.relu(ligand_x)
        ligand_x = self.ligand_conv2(ligand_x, ligand_data.edge_index)
        ligand_x = torch.nn.functional.relu(ligand_x)
        
        # Global pooling to get graph-level representations
        protein_repr = torch_geometric.nn.global_mean_pool(protein_x, protein_data.batch)
        ligand_repr = torch_geometric.nn.global_mean_pool(ligand_x, ligand_data.batch)
        
        # Predict binding affinity
        combined = torch.cat([protein_repr, ligand_repr], dim=1)
        binding_affinity = self.interaction(combined)
        
        return binding_affinity
```

### Training Strategy
- **Loss function selection**:
  - Mean Squared Error (MSE) for regression
  - Concordance Index (CI) as a ranking metric
  - Custom functions combining regression and ranking

- **Optimization considerations**:
  - Learning rate scheduling (1e-4 to 1e-5 typical)
  - Batch size impacts (16-128 depending on model)
  - Early stopping based on validation performance
  - Gradient clipping for stability

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

```python
# Example evaluation code
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from scipy.stats import pearsonr, spearmanr

def evaluate_binding_model(model, test_loader, device):
    model.eval()
    true_values = []
    predicted_values = []
    
    with torch.no_grad():
        for protein_data, ligand_data, affinity in test_loader:
            protein_data = protein_data.to(device)
            ligand_data = ligand_data.to(device)
            
            prediction = model(protein_data, ligand_data)
            
            true_values.extend(affinity.cpu().numpy())
            predicted_values.extend(prediction.cpu().numpy())
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    r2 = r2_score(true_values, predicted_values)
    pearson = pearsonr(true_values, predicted_values)[0]
    spearman = spearmanr(true_values, predicted_values)[0]
    
    # Calculate concordance index
    ci = concordance_index(true_values, predicted_values)
    
    return {
        "RMSE": rmse,
        "R²": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "CI": ci
    }

def concordance_index(y_true, y_pred):
    """Calculate concordance index for binding affinity prediction"""
    pairs = 0
    concordant = 0
    
    for i in range(len(y_true)):
        for j in range(len(y_true)):
            if i != j and y_true[i] != y_true[j]:
                pairs += 1
                if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
                   (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                    concordant += 1
    
    return concordant / pairs if pairs > 0 else 0
```

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
"""
        ),
        FewShotExample(
            query="How do I use deep learning for variant calling in next-generation sequencing data?",
            context="The user has whole genome sequencing data and wants to apply deep learning approaches to improve variant detection accuracy.",
            response="""
# Deep Learning for Variant Calling in Genomic Data

## Approach Overview
Applying deep learning to variant calling involves:
1. Understanding traditional vs. AI-based variant calling approaches
2. Preparing appropriate training data
3. Selecting suitable model architectures
4. Implementing effective training strategies
5. Evaluating and fine-tuning your model

## Key Decision Points

### Variant Calling Approach Selection
Consider these approaches based on your needs:

- **End-to-end deep learning approaches**:
  - Completely replace traditional variant callers
  - Examples: DeepVariant, NeuSomatic
  - Suitable when maximum accuracy is the goal

- **Hybrid approaches**:
  - Use ML/DL to refine calls from traditional methods
  - Examples: GATK CNNScoreVariants, GARFIELD-NGS
  - Good balance of speed and accuracy

- **Ensemble methods**:
  - Combine multiple callers with ML/DL integration
  - Examples: SomaticSeq, SMuRF
  - Particularly effective for somatic variants

- **Custom approach development**:
  - Build specialized models for specific variant types
  - Incorporate unique data sources
  - Adapt to specific sequencing technologies

### Data Preparation and Preprocessing
- **Input representation options**:
  - Pileup images (DeepVariant approach)
  - Tensor representations of read alignments
  - Feature vectors from BAM files
  - Raw signal data (for Nanopore)

- **Training dataset considerations**:
  - Synthetic/simulated variants vs. validated variants
  - Genome in a Bottle (GIAB) benchmark datasets
  - COSMIC/TCGA for somatic variants
  - Importance of including difficult regions

- **Preprocessing decisions**:
  - Read alignment filters (mapping quality, duplicates)
  - Candidate variant selection methods
  - Balancing variant/non-variant examples
  - Channel selection for tensor-based models

```python
# Example of creating a pileup image dataset (conceptual)
import pysam
import numpy as np
from PIL import Image

def create_pileup_image(bam_file, reference_file, chrom, position, window_size=100):
    """Create a pileup image centered on a specific genomic position"""
    # Initialize pileup matrix (row for each read, column for each position)
    # Channels: base (A,C,G,T), quality, strand, etc.
    pileup_tensor = np.zeros((100, window_size, 8), dtype=np.uint8)
    
    # Open BAM and reference files
    bam = pysam.AlignmentFile(bam_file, "rb")
    ref = pysam.FastaFile(reference_file)
    
    # Get reference sequence for the window
    ref_seq = ref.fetch(chrom, position-window_size//2, position+window_size//2)
    
    # Fill tensor based on reads in the region
    read_idx = 0
    for read in bam.fetch(chrom, position-window_size//2, position+window_size//2):
        if read_idx >= 100:  # Limit to 100 reads
            break
            
        # Process the read and update pileup_tensor
        # [code to fill tensor with base information, quality scores, etc.]
        
        read_idx += 1
    
    # Convert tensor to RGB image for visualization/input to CNN
    pileup_image = np.zeros((100, window_size, 3), dtype=np.uint8)
    # [code to convert tensor channels to RGB representation]
    
    return pileup_tensor, pileup_image
```

### Model Architecture Selection
- **CNN-based architectures**:
  - Inception-based (DeepVariant)
  - ResNet/DenseNet adaptations
  - Good for pileup images and spatial patterns

- **RNN/LSTM approaches**:
  - Capture sequential information in reads
  - Handle variable-length inputs
  - Good for modeling sequence context

- **Transformer-based models**:
  - Attention to important read positions
  - Capture long-range dependencies
  - Examples: NeuSomatic, Longshot

- **Technology-specific considerations**:
  - Signal-based models for Nanopore (e.g., Medaka)
  - Read length-aware architectures for long reads
  - Depth-adaptive models for varying coverage

### Training Strategy
- **Loss function selection**:
  - Binary/categorical cross-entropy for variant classification
  - Focal loss for imbalanced datasets
  - Custom loss incorporating variant quality

- **Class imbalance handling**:
  - Weighted loss functions
  - Negative sampling strategies
  - Data augmentation for rare variant types

- **Transfer learning opportunities**:
  - Pre-training on high-confidence regions
  - Fine-tuning for specific genomic contexts
  - Cross-species adaptation

- **Validation strategy**:
  - Chromosome-based splits (avoid read contamination)
  - Stratified sampling across variant types
  - Separate validation for different genomic contexts

```python
# Example of a custom loss function for variant calling
import tensorflow as tf

def variant_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss adapted for variant calling with additional weighting for different variant types
    """
    # Convert one-hot encoded y_true to class indices
    y_true_class = tf.argmax(y_true, axis=-1)
    
    # Extract variant type weights from y_true (e.g., SNV, indel, etc.)
    variant_weights = tf.gather(tf.constant([1.0, 2.0, 3.0, 5.0]), y_true_class)
    
    # Compute focal loss
    cross_entropy = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
    probs = tf.reduce_sum(y_true * y_pred, axis=-1)
    focal_weight = alpha * tf.pow(1.0 - probs, gamma)
    
    # Combine focal weight with variant type weight
    combined_weight = focal_weight * variant_weights
    
    return combined_weight * cross_entropy
```

### Evaluation and Refinement
- **Key evaluation metrics**:
  - Precision/Recall/F1 by variant type
  - ROC and Precision-Recall curves
  - Genotype concordance
  - Stratification by genomic region types

- **Filtering and post-processing**:
  - Quality score recalibration
  - Ensemble integration
  - Region-aware filtering

- **Systematic error analysis**:
  - Confusion patterns by variant type
  - Genomic context of errors
  - Coverage/quality correlation with errors

## Interpretation Considerations

### Performance Assessment Contexts
- **Comparison frameworks**:
  - RTG vcfeval
  - Genome in a Bottle benchmarking
  - GA4GH standardized variant comparisons
  - hap.py/som.py evaluation frameworks

- **Variant stratification**:
  - By type: SNVs, indels, structural variants
  - By region: high/low complexity, repetitive regions
  - By allele frequency
  - By genomic feature (exon, intron, regulatory)

```python
# Example code for stratified variant evaluation
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def evaluate_by_region(truth_vcf, test_vcf, region_beds):
    """Evaluate variant calling performance stratified by genomic regions"""
    results = []
    
    # First, use rtg vcfeval or hap.py to get TP, FP, FN variants
    # [code to run comparison tool and parse results]
    
    all_tp = get_true_positives(truth_vcf, test_vcf)
    all_fp = get_false_positives(truth_vcf, test_vcf)
    all_fn = get_false_negatives(truth_vcf, test_vcf)
    
    # Overall metrics
    precision = len(all_tp) / (len(all_tp) + len(all_fp)) if len(all_tp) + len(all_fp) > 0 else 0
    recall = len(all_tp) / (len(all_tp) + len(all_fn)) if len(all_tp) + len(all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    results.append({
        'region': 'overall',
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': len(all_tp),
        'fp': len(all_fp),
        'fn': len(all_fn)
    })
    
    # Stratify by region
    for region_name, region_bed in region_beds.items():
        # Filter variants by region
        region_tp = filter_by_region(all_tp, region_bed)
        region_fp = filter_by_region(all_fp, region_bed)
        region_fn = filter_by_region(all_fn, region_bed)
        
        # Calculate metrics
        region_precision = len(region_tp) / (len(region_tp) + len(region_fp)) if len(region_tp) + len(region_fp) > 0 else 0
        region_recall = len(region_tp) / (len(region_tp) + len(region_fn)) if len(region_tp) + len(region_fn) > 0 else 0
        region_f1 = 2 * region_precision * region_recall / (region_precision + region_recall) if region_precision + region_recall > 0 else 0
        
        results.append({
            'region': region_name,
            'precision': region_precision,
            'recall': region_recall,
            'f1': region_f1,
            'tp': len(region_tp),
            'fp': len(region_fp),
            'fn': len(region_fn)
        })
    
    return pd.DataFrame(results)
```

### Special Considerations
- **Sequencing technology-specific challenges**:
  - Illumina: PCR errors, GC bias
  - PacBio: Lower error rates but still indel-prone
  - Nanopore: Homopolymer regions, methylation effects

- **Challenging variant types**:
  - Complex indels
  - Variants in repetitive regions
  - Low-frequency somatic variants
  - Structural variants

- **Computational resource management**:
  - Most DL models are GPU-intensive
  - Batch size optimization for memory constraints
  - Inference time considerations for production use

### Emerging Trends
- **Multi-modal integration**:
  - Combining short and long reads
  - Adding methylation data
  - Haplotype-aware variant calling

- **Adaptive models**:
  - Sample-specific calibration
  - Coverage-adaptive architectures
  - Online learning approaches

- **Specialized approaches**:
  - Somatic variant-specific models
  - Rare disease variant prioritization
  - Population-specific genetic variation modeling
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

```python
# Example of RNA-seq preprocessing and batch correction
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scanpy as sc

def preprocess_rnaseq(counts_matrix, metadata, log_transform=True, batch_correct=True):
    """Preprocess RNA-seq data for machine learning"""
    # Load count matrix
    expression = pd.DataFrame(counts_matrix)
    
    # Basic filtering
    # Remove genes with low counts across most samples
    min_counts = 10
    min_samples = 5
    genes_to_keep = ((expression > min_counts).sum(axis=1) >= min_samples)
    filtered_expr = expression.loc[genes_to_keep]
    
    # Log2 transformation (with pseudocount)
    if log_transform:
        filtered_expr = np.log2(filtered_expr + 1)
    
    # Batch correction if needed
    if batch_correct and 'batch' in metadata.columns:
        # Convert to AnnData for scanpy-based batch correction
        adata = sc.AnnData(filtered_expr.T)
        adata.obs['batch'] = metadata['batch'].values
        
        # Apply batch correction (Combat)
        sc.pp.combat(adata, key='batch')
        corrected_expr = pd.DataFrame(adata.X.T, index=filtered_expr.index, columns=filtered_expr.columns)
    else:
        corrected_expr = filtered_expr
    
    # Scale the data
    scaler = StandardScaler()
    scaled_expr = pd.DataFrame(
        scaler.fit_transform(corrected_expr.T).T,
        index=corrected_expr.index,
        columns=corrected_expr.columns
    )
    
    return scaled_expr
```

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

```python
# Example of dimensionality reduction and visualization
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from umap import UMAP

def reduce_and_visualize(expression_data, labels, n_genes=2000):
    """Apply dimensionality reduction and visualize cancer subtypes"""
    # Select top variable genes
    gene_variance = expression_data.var(axis=1)
    top_variable_genes = gene_variance.sort_values(ascending=False).head(n_genes).index
    expr_subset = expression_data.loc[top_variable_genes]
    
    # Apply PCA
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(expr_subset.T)
    
    # Apply UMAP on PCA results
    umap_model = UMAP(n_neighbors=30, min_dist=0.3, metric='correlation')
    umap_result = umap_model.fit_transform(pca_result)
    
    # Visualize
    plt.figure(figsize=(12, 10))
    
    # Create color map based on unique cancer subtypes
    unique_subtypes = list(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_subtypes)))
    color_map = dict(zip(unique_subtypes, colors))
    
    # Plot each subtype
    for subtype in unique_subtypes:
        mask = [l == subtype for l in labels]
        plt.scatter(
            umap_result[mask, 0],
            umap_result[mask, 1],
            s=50, 
            alpha=0.7,
            c=[color_map[subtype]],
            label=subtype
        )
    
    plt.legend(markerscale=2)
    plt.title('UMAP projection of gene expression data', fontsize=14)
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    
    # Variance explained by PCA
    plt.figure(figsize=(10, 4))
    plt.bar(range(1, 21), pca.explained_variance_ratio_[:20] * 100)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained (%)')
    plt.title('PCA Variance Explained')
    plt.tight_layout()
    
    return pca_result, umap_result, pca.explained_variance_ratio_
```

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

```python
# Example of model selection and training with cross-validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, balanced_accuracy_score

def train_cancer_classifier(X, y, n_splits=5):
    """Train and evaluate multiple classifiers for cancer subtype prediction"""
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize results dictionary
    results = {}
    
    # 1. Random Forest
    rf_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1
    )
    
    rf_grid.fit(X, y)
    results['random_forest'] = {
        'best_params': rf_grid.best_params_,
        'best_score': rf_grid.best_score_,
        'model': rf_grid.best_estimator_
    }
    
    # 2. Implement additional models (SVM, Gradient Boosting, etc.)
    # [code for additional models]
    
    # Evaluate best model more thoroughly
    best_model = results['random_forest']['model']  # Assuming RF is best
    y_pred = best_model.predict(X)
    
    print(classification_report(y, y_pred))
    
    # Feature importance for interpretability
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        sorted_idx = np.argsort(feature_importance)[::-1]
        top_features = sorted_idx[:20]  # Top 20 features
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(range(20), feature_importance[top_features], align='center')
        plt.yticks(range(20), [feature_names[i] for i in top_features])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Important Features')
        plt.tight_layout()
    
    return results
```

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

```python
# Example of robust evaluation with nested cross-validation
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def robust_model_evaluation(X, y, model, cv_outer=5, cv_inner=3):
    """Perform nested cross-validation with proper evaluation metrics"""
    # Outer CV for performance estimation
    cv_outer_split = StratifiedKFold(n_splits=cv_outer, shuffle=True, random_state=42)
    
    # Arrays to store results
    cv_scores = []
    all_y_true = []
    all_y_pred = []
    all_probas = []
    
    # For each outer split
    for train_idx, test_idx in cv_outer_split.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for hyperparameter tuning
        inner_cv = StratifiedKFold(n_splits=cv_inner, shuffle=True, random_state=42)
        
        # Clone and fit model
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        
        # Calibrate probabilities
        calibrated_model = CalibratedClassifierCV(model_clone, cv='prefit')
        calibrated_model.fit(X_train, y_train)
        
        # Predict
        y_pred = calibrated_model.predict(X_test)
        y_proba = calibrated_model.predict_proba(X_test)
        
        # Calculate score
        score = balanced_accuracy_score(y_test, y_pred)
        cv_scores.append(score)
        
        # Store for later analysis
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_probas.append(y_proba)
    
    # Overall performance
    print(f"Mean balanced accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
    disp.plot(cmap='Blues')
    plt.title('Normalized Confusion Matrix')
    
    # Additional evaluations (ROC curves, calibration plots, etc.)
    # [code for additional visualizations]
    
    return {
        'cv_scores': cv_scores,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
        'y_proba': all_probas
    }
```

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

```python
# Example for choosing the right approach based on protein characteristics
def select_structure_prediction_approach(sequence, has_homologs=True, is_multimeric=False, 
                                        is_membrane=False, compute_resources="medium"):
    """Select appropriate protein structure prediction approach based on characteristics"""
    
    # Analyze sequence properties
    length = len(sequence)
    is_large = length > 1000
    
    # Check for available homologs
    has_close_homologs = check_close_homologs(sequence) if has_homologs else False
    
    # Recommend approaches
    if compute_resources == "limited":
        if length < 400:
            return "ESMFold (faster but less accurate)"
        else:
            return "ColabFold with reduced iterations"
    
    if is_multimeric:
        if is_large:
            return "AlphaFold-Multimer with domain splitting"
        else:
            return "AlphaFold-Multimer or RoseTTAFold-Complex"
    
    if is_membrane:
        if has_close_homologs:
            return "AlphaFold2 with membrane-specific templates"
        else:
            return "MemBrain + AlphaFold2"
    
    # Default for soluble proteins
    if is_large:
        return "AlphaFold2 with domain splitting and assembly"
    else:
        return "AlphaFold2 or RoseTTAFold full pipeline"
```

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

```python
# Example of MSA generation for structure prediction
import subprocess
import os

def generate_msas_for_structure_prediction(sequence, seq_id, work_dir="./"):
    """Generate MSAs using multiple search tools for optimal results"""
    # Create directories
    os.makedirs(f"{work_dir}/msas", exist_ok=True)
    
    # Write sequence to FASTA
    with open(f"{work_dir}/{seq_id}.fasta", "w") as f:
        f.write(f">{seq_id}\n{sequence}\n")
    
    # Run different search tools in parallel
    # 1. HHblits against UniClust30
    hhblits_cmd = [
        "hhblits",
        "-i", f"{work_dir}/{seq_id}.fasta",
        "-o", f"{work_dir}/msas/{seq_id}_hhblits.hhr",
        "-oa3m", f"{work_dir}/msas/{seq_id}_hhblits.a3m",
        "-d", "UniClust30_2018_08/UniClust30_2018_08",
        "-n", "3",
        "-e", "0.001",
        "-maxfilt", "100000",
        "-neffmax", "20",
        "-nodiff"
    ]
    
    # 2. Jackhmmer against UniRef90
    jackhmmer_cmd = [
        "jackhmmer",
        "--cpu", "8",
        "-N", "3",
        "--incE", "0.0001",
        "--noali",
        "--tblout", f"{work_dir}/msas/{seq_id}_jackhmmer.tbl",
        "-A", f"{work_dir}/msas/{seq_id}_jackhmmer.sto",
        f"{work_dir}/{seq_id}.fasta",
        "uniref90.fasta"
    ]
    
    # 3. MMseqs2 for faster search against BFD
    mmseqs_cmd = [
        "mmseqs", "easy-search",
        f"{work_dir}/{seq_id}.fasta",
        "bfd_db",
        f"{work_dir}/msas/{seq_id}_mmseqs.m8",
        f"{work_dir}/msas/tmp",
        "--format-output", "query,target,evalue,gapopen,pident,nident,qstart,qend,tstart,tend,qlen,tlen,alnlen,raw,bits,cigar,qseq,tseq",
        "-s", "7.5",
        "--num-iterations", "3",
        "--slice-search"
    ]
    
    # Run searches and parse results
    processes = [
        subprocess.Popen(hhblits_cmd),
        subprocess.Popen(jackhmmer_cmd),
        subprocess.Popen(mmseqs_cmd)
    ]
    
    # Wait for all processes to complete
    for p in processes:
        p.wait()
    
    # Convert to formats needed for AlphaFold/RoseTTAFold
    # [Additional code for format conversion...]
    
    return {
        "hhblits_a3m": f"{work_dir}/msas/{seq_id}_hhblits.a3m",
        "jackhmmer_sto": f"{work_dir}/msas/{seq_id}_jackhmmer.sto",
        "mmseqs_m8": f"{work_dir}/msas/{seq_id}_mmseqs.m8"
    }
```

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

```python
# Example of structure evaluation and quality assessment
from Bio.PDB import PDBParser
import py3Dmol
import matplotlib.pyplot as plt
import numpy as np

def evaluate_predicted_structure(predicted_pdb, reference_pdb=None):
    """Evaluate and visualize a predicted protein structure"""
    # Parse structures
    parser = PDBParser()
    predicted_structure = parser.get_structure("predicted", predicted_pdb)
    
    # Extract pLDDT values if available (from B-factor column in AF2 outputs)
    residue_confidence = []
    for residue in predicted_structure[0].get_residues():
        for atom in residue:
            if atom.name == "CA":
                residue_confidence.append(atom.bfactor)
    
    # Plot confidence profile
    plt.figure(figsize=(10, 4))
    plt.plot(residue_confidence)
    plt.xlabel('Residue position')
    plt.ylabel('pLDDT confidence')
    plt.title('Prediction confidence by residue')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=90, color='g', linestyle='--')
    plt.fill_between(range(len(residue_confidence)), 
                     residue_confidence, 
                     90, 
                     where=[x >= 90 for x in residue_confidence], 
                     color='green', alpha=0.3)
    plt.fill_between(range(len(residue_confidence)), 
                     residue_confidence, 
                     70, 
                     where=[70 <= x < 90 for x in residue_confidence], 
                     color='yellow', alpha=0.3)
    plt.fill_between(range(len(residue_confidence)), 
                     residue_confidence, 
                     50, 
                     where=[50 <= x < 70 for x in residue_confidence], 
                     color='orange', alpha=0.3)
    plt.fill_between(range(len(residue_confidence)), 
                     residue_confidence, 
                     0, 
                     where=[x < 50 for x in residue_confidence], 
                     color='red', alpha=0.3)
    
    # Calculate global metrics if reference is available
    if reference_pdb:
        reference_structure = parser.get_structure("reference", reference_pdb)
        # [Code to calculate TM-score, RMSD, GDT-TS using external tools]
        
    # Visualize structure colored by confidence
    view = py3Dmol.view(width=800, height=500)
    view.addModel(open(predicted_pdb, 'r').read(), 'pdb')
    view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min': 50, 'max': 90}}})
    view.zoomTo()
    
    return {
        'mean_plddt': np.mean(residue_confidence),
        'high_confidence_regions': [i for i, conf in enumerate(residue_confidence) if conf >= 70],
        'low_confidence_regions': [i for i, conf in enumerate(residue_confidence) if conf < 50],
        'visualization': view
    }
```

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
    with open("artificial_intelligence_prompt.json", "w") as f:
        f.write(artificial_intelligence_prompt.to_json())