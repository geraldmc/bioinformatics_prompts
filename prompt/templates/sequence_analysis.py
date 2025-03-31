"""Module containing the sequence analysis research_area prompt template."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a sequence analysis prompt template
sequence_analysis_prompt = BioinformaticsPrompt(
    research_area="Sequence Analysis",
    description=(
        "Sequence analysis is a fundamental aspect of bioinformatics that involves the examination of DNA, RNA, or protein "
        "sequences to derive meaningful biological information. It encompasses a wide range of techniques, including sequence "
        "alignment, motif discovery, sequence assembly, variant detection, and phylogenetic analysis. With the evolution of "
        "sequencing technologies from short to long reads, the computational methods for sequence analysis continue to adapt "
        "to provide more comprehensive insights into genomic data."
    ),
    key_concepts=[
        "Global and local sequence alignment",
        "Pairwise and multiple sequence alignment",
        "Sequence homology and similarity",
        "Motif discovery and enrichment analysis",
        "Phylogenetic tree construction and analysis",
        "Long-read sequencing analysis (PacBio, Oxford Nanopore)",
        "Sequence assembly (de novo and reference-guided)",
        "Consensus sequence generation",
        "Open reading frame (ORF) prediction",
        "Sequence annotation and functional prediction"
    ],
    common_tools=[
        "BLAST/DIAMOND - Sequence similarity search",
        "MAFFT/Clustal Omega/T-Coffee - Multiple sequence alignment",
        "HMMER - Profile hidden Markov models for sequence analysis",
        "MEME Suite - Motif discovery and analysis",
        "IQ-TREE/RAxML/FastTree - Phylogenetic inference",
        "Minimap2 - Long-read alignment",
        "Canu/Flye - Long-read assembly",
        "MEGA - Molecular evolutionary genetics analysis",
        "EMBOSS - Sequence analysis package",
        "CD-HIT - Sequence clustering"
    ],
    common_file_formats=[
        {"name": "FASTA", "description": "Nucleotide or protein sequences without quality scores"},
        {"name": "FASTQ", "description": "Nucleotide sequences with quality scores"},
        {"name": "CLUSTAL/STOCKHOLM", "description": "Multiple sequence alignment formats"},
        {"name": "NEWICK/NEXUS/PHYLIP", "description": "Phylogenetic tree formats"},
        {"name": "GenBank/EMBL/GFF", "description": "Annotated sequence formats"},
        {"name": "BAM/SAM", "description": "Sequence alignment formats"},
        {"name": "VCF", "description": "Variant call format"}
    ],
    examples=[
        FewShotExample(
            query="How do I perform phylogenetic analysis for a set of protein sequences?",
            context="The user has 50 homologous protein sequences in FASTA format from different bacterial species and wants to understand their evolutionary relationships.",
            response="""
# Phylogenetic Analysis of Bacterial Protein Sequences

## Approach Overview
For phylogenetic analysis of bacterial protein sequences, you'll need to:
1. Align the sequences
2. Select an appropriate evolutionary model
3. Build a phylogenetic tree
4. Evaluate tree reliability

## Key Decision Points

### Multiple Sequence Alignment (MSA)
- **For <200 sequences with moderate divergence**: Use MAFFT with L-INS-i (--localpair --maxiterate 1000)
- **For larger datasets**: Use MAFFT with FFT-NS-2 (default) or MUSCLE
- **For highly divergent sequences**: Consider T-Coffee or PROMALS3D which can incorporate structural information

The accuracy of your alignment is critical - always visualize and inspect it before proceeding.

```bash
# Example for medium-sized datasets with moderate divergence
mafft --localpair --maxiterate 1000 --reorder protein_sequences.fasta > aligned_proteins.fasta
```

### Trimming Alignments
Removing poorly aligned regions can improve tree inference:
- **Automated trimming**: Use trimAl or Gblocks
- **Manual trimming**: For smaller datasets, use Jalview or AliView

Whether to trim depends on your sequences - conserved proteins may not need trimming, while divergent sequences benefit from it.

### Evolutionary Model Selection
The choice of evolutionary model significantly impacts tree accuracy:
- **Automated selection**: Use ModelFinder (in IQ-TREE) or ProtTest
- **Common protein models**: LG, WAG, or JTT with rate heterogeneity (+G)
- **For prokaryotic proteins**: LG+G is often a good starting point

```bash
# Model selection with IQ-TREE
iqtree -s aligned_proteins.fasta -m MFP -mfreq -msub nuclear
```

### Tree Building Method
Different methods have different strengths:
- **Maximum Likelihood**: IQ-TREE or RAxML (most commonly used for modern analyses)
- **Bayesian Inference**: MrBayes or BEAST (when prior information is available)
- **Distance-Based**: FastME or Neighbor-Joining (for quick initial trees)

For bacterial proteins, Maximum Likelihood with IQ-TREE offers an excellent balance of accuracy and speed.

```bash
# Maximum Likelihood tree with IQ-TREE
iqtree -s aligned_proteins.fasta -m LG+G4 -bb 1000 -alrt 1000
```

### Tree Reliability Assessment
Always evaluate support for your tree topology:
- **Bootstrap support**: Values >70% indicate reliable branches
- **SH-aLRT test**: Provides faster assessment than traditional bootstrapping
- **Ultrafast Bootstrap (UFBoot)**: Faster than standard bootstrapping but may be overconfident

Assess the tree with multiple approaches when possible. For publication-quality trees, 1000 bootstrap replicates are typically expected.

## Interpretation Considerations
- Branches with low support (<70% bootstrap) should be interpreted with caution
- Consider gene-specific evolutionary rates when interpreting branch lengths
- Horizontal gene transfer in bacteria may cause individual gene trees to differ from species trees
- Outgroup selection can significantly affect tree topology - choose an appropriate outgroup that is not too distant

## Visualization
For effective tree visualization:
- **Interactive exploration**: Use FigTree, iTOL, or Dendroscope
- **Publication figures**: Use ggtree (in R) for customizable, publication-quality images
- **Annotating trees**: Add metadata like habitat, pathogenicity, or geographic origin

## When to Consider Alternative Approaches
- **For very large datasets** (>1000 sequences): Consider FastTree or IQ-TREE with fast models
- **For highly divergent sequences**: Consider using protein domains rather than full sequences
- **For closely related sequences**: Use nucleotide sequences instead of proteins
- **For dataset with insertion/deletion bias**: Consider phylogenetic placement instead of de novo tree building
"""
        ),
        FewShotExample(
            query="What approach should I use to identify protein motifs in a set of sequences?",
            context="The user has extracted 200 protein sequences that are co-expressed in response to stress and wants to identify shared functional motifs.",
            response="""
# Identifying Protein Motifs in Stress-Response Proteins

## Approach Overview
To identify functional motifs in stress-response proteins, you'll need to:
1. Prepare and filter your sequence dataset
2. Perform de novo motif discovery
3. Compare with known motifs
4. Validate motifs with structural and functional analysis

## Key Decision Points

### Dataset Preparation
- **Sequence redundancy**: Remove highly similar sequences (typically >90% identity)
- **Sequence length**: Group by similar lengths or domains
- **Background sequences**: Select appropriate negative dataset (non-stress-responsive proteins)

```python
# Example of redundancy reduction using CD-HIT
from Bio import SeqIO
import subprocess

# Write sequences to file
SeqIO.write(sequences, "stress_proteins.fasta", "fasta")

# Run CD-HIT with 90% identity threshold
subprocess.run(["cd-hit", "-i", "stress_proteins.fasta", "-o", "nr_proteins.fasta", "-c", "0.9"])
```

### De Novo Motif Discovery
Choose the right algorithm based on your expectations about the motifs:

- **For short, ungapped motifs** (e.g., transcription factor binding sites):
  - MEME (with -mod zoops or -mod anr)
  - STREME (successor to DREME, for short motifs)

- **For longer, structured motifs** (e.g., enzyme active sites):
  - GLAM2 (allows for insertions/deletions)
  - HMMER (if you have an initial alignment of the motif region)

- **For periodic or spaced motifs**:
  - MEME with appropriate gap parameters
  - MAST for scanning with complex motif models

```bash
# Example for discovering ungapped motifs with MEME
meme nr_proteins.fasta -protein -oc meme_output -nmotifs 5 -minw 6 -maxw 50 -mod zoops
```

### Comparing with Known Motifs
After discovering motifs, determine if they're novel or known:

- **For known motif comparison**:
  - Search against ELM (Eukaryotic Linear Motif) database for short functional motifs
  - Search PROSITE or Pfam for domain signatures
  - Use Tomtom to compare against motif databases

- **Parameters to consider**:
  - E-value thresholds (typically 0.05 for initial screening)
  - Query coverage (how much of your motif matches the database entry)
  - Conservation across species (conserved motifs are more likely functional)

```bash
# Example of comparing discovered motifs to known databases
tomtom -oc tomtom_output -min-overlap 5 -dist pearson -thresh 0.05 meme_output/meme.xml motif_database.meme
```

### Structural Context Analysis
Evaluate the structural context of your motifs:

- **For proteins with known structures**:
  - Map motifs onto 3D structures using PyMOL or UCSF Chimera
  - Check if motifs are in structured domains or disordered regions
  
- **For proteins without structures**:
  - Use structure prediction tools like AlphaFold2
  - Predict disordered regions using IUPred or PONDR
  - Predict secondary structure using PSIPRED

- **Key features to look for**:
  - Surface accessibility (functional motifs often surface-exposed)
  - Conservation of 3D arrangement in different proteins
  - Proximity to active sites or binding interfaces

## Interpretation Considerations

### Biological Validation Strategies
Consider how to validate predicted motifs:
- Functional assays with site-directed mutagenesis
- Peptide array binding experiments
- Evolutionary conservation analysis across diverse species

### Motif Classification
Based on your analysis, classify motifs by:

- **Function**:
  - Catalytic motifs (e.g., active site signatures)
  - Binding motifs (e.g., protein-protein interaction sites)
  - PTM sites (e.g., phosphorylation motifs)
  - Localization signals (e.g., nuclear localization sequences)

- **Structure**:
  - Secondary structure elements (α-helices, β-strands)
  - Disordered region motifs (often linear motifs)
  - Metal-binding motifs (e.g., zinc fingers)

### Stress-Response Specific Considerations
For stress-response proteins specifically:
- Look for known stress-related motifs (heat shock elements, oxidative stress response elements)
- Consider condition-specific motifs that may be activated only under specific stressors
- Examine motif conservation across different stress conditions
- Compare with motifs in proteins with similar function but different regulation
"""
        ),
        FewShotExample(
            query="How do I analyze long-read sequencing data for detecting structural variants?",
            context="The user has Oxford Nanopore whole genome sequencing data from a human sample and wants to identify structural variants.",
            response="""
# Structural Variant Detection with Long-Read Sequencing

## Approach Overview
To detect structural variants (SVs) using Oxford Nanopore long-read data:
1. Prepare and QC your sequencing data
2. Align reads to a reference genome
3. Apply SV detection algorithms
4. Filter and validate candidate SVs

## Key Decision Points

### Data Quality Assessment
- **Read length distribution**: Longer reads (>10kb) improve SV detection
- **Coverage requirements**: 
  - Minimum 15-20x for large SVs (>1kb)
  - 30x+ recommended for comprehensive detection
  - 50x+ for detecting low-frequency somatic variants
- **Base quality**: Consider re-basecalling with latest models if using older data

```python
# Example code snippet to check read length distribution
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO

# Load read lengths
lengths = [len(record) for record in SeqIO.parse("reads.fastq", "fastq")]

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=50, alpha=0.75)
plt.title("Read Length Distribution")
plt.xlabel("Read Length (bp)")
plt.ylabel("Frequency")
plt.axvline(np.median(lengths), color='r', linestyle='dashed', linewidth=1)
plt.text(np.median(lengths)*1.1, plt.ylim()[1]*0.9, f"Median: {np.median(lengths):.0f} bp")
plt.savefig("read_length_dist.png")
```

### Alignment Strategy
- **Aligner selection**:
  - Minimap2 (most widely used, excellent balance of speed and accuracy)
  - NGMLR (slower but may perform better for certain SV types)
  - LAST (older but sometimes better for highly divergent sequences)

- **Alignment parameters**:
  - For human genomes, use the "map-ont" preset in Minimap2
  - Adjust mapping quality thresholds based on your data quality
  - Consider masking problematic regions (e.g., centromeres, telomeres)

### SV Detection Tools
Different SV callers have different strengths - use multiple for comprehensive detection:

- **Primary SV callers**:
  - Sniffles2 (excellent all-around performance)
  - SVIM (good sensitivity, especially for insertions)
  - cuteSV (high performance, good for complex SVs)
  - NanoSV (specifically designed for ONT data)

- **SV caller selection based on variant type**:
  - For deletions: Most callers perform similarly well
  - For insertions: SVIM and Sniffles2 typically perform best
  - For inversions and translocations: Consider specialized callers or manual verification

### SV Filtering and Merging
- **Filtering parameters**:
  - Read support (typically ≥3-5 reads for germline variants)
  - Mapping quality (usually ≥20)
  - SV length (based on your research question)
  - Proximity to difficult regions

- **Multi-caller approach**:
  - Use SURVIVOR or SVDB to merge calls from multiple tools
  - Require SV detection by at least 2 independent methods
  - Allow for positioning differences (typically 500-1000bp for large SVs)

```bash
# Example SURVIVOR command (conceptual)
SURVIVOR merge callsets.txt 1000 2 1 1 0 50 merged_SVs.vcf
# Parameters: max distance (1000bp), min callers (2), consider type/strands/regions (1/1/0), min size (50bp)
```

## Interpretation Considerations

### SV Classification
Understand the different classes of SVs:
- **Deletions**: Usually the most reliably detected
- **Insertions**: Challenging due to reference bias, but long reads excel here
- **Duplications**: Look for read depth evidence in addition to split reads
- **Inversions**: Require reads spanning both breakpoints
- **Translocations**: High false positive rate, require careful validation

### Validating SVs
Consider these validation approaches:
- **Orthogonal technologies**: PCR, optical mapping, linked-reads
- **Manual inspection**: IGV visualization of discordant reads and split alignments
- **Population frequency**: Check databases like gnomAD-SV for known variants
- **Local assembly**: Assemble reads around breakpoints for precise SV characterization

### Long-Read Specific Advantages
Leverage the unique strengths of long reads:
- The ability to span entire SVs and complex regions
- Better resolution of breakpoints
- Detection of insertions and novel sequences
- Resolution of SVs in repetitive regions

### Known Challenges
Be aware of these common issues:
- Tandem repeats and segmental duplications remain challenging
- Alignment errors in low-complexity regions
- Higher error rates can affect precise breakpoint detection
- Limited databases of known SVs compared to SNVs

## Follow-up Analysis
After identifying SVs, consider:
- Annotating SVs with gene impacts using AnnotSV or SnpEff
- Pathway enrichment analysis for affected genes
- Comparing SVs across related samples
- Correlating SVs with phenotypic data if available
"""
        ),
        FewShotExample(
            query="What are the key considerations for de novo genome assembly with long reads?",
            context="The user is planning an Oxford Nanopore sequencing project to assemble a novel bacterial genome and wants to understand the key decisions in the assembly process.",
            response="""
# De Novo Genome Assembly with Long Reads

## Approach Overview
For de novo assembly of a bacterial genome with Oxford Nanopore long reads:
1. Plan your sequencing strategy
2. Process and assess read quality
3. Select appropriate assembly methods
4. Polish and validate your assembly

## Key Decision Points

### Sequencing Strategy
- **Coverage requirements**:
  - For bacterial genomes: 50-100x for high-quality assembly
  - Higher coverage compensates for lower accuracy of individual reads
  - Diminishing returns above ~100x coverage
  
- **Read length considerations**:
  - Longer reads (>20kb) improve contiguity dramatically
  - Library prep methods affect read length (ligation > rapid > transposase)
  - Ultra-long reads (>100kb) can span most bacterial repeats
  
- **DNA extraction**:
  - High molecular weight extraction methods preserve long fragments
  - Avoid excessive shearing during extraction
  - Consider specialized kits for difficult-to-lyse bacteria

### Quality Control and Preprocessing
- **Read filtering decisions**:
  - Quality score threshold (typically Q7-Q10)
  - Minimum read length (typically 1-5kb)
  - Adaptor trimming and chimera removal
  
- **Contamination screening**:
  - BLAST-based classification of reads
  - Kraken2/Centrifuge for metagenomic screening
  - Identification of host contamination

```python
# Example: Quick assessment of potential read contamination
from Bio import SeqIO
import subprocess
import random

# Randomly sample reads for faster processing
sampled_reads = random.sample([rec for rec in SeqIO.parse("reads.fastq", "fastq")], 1000)
SeqIO.write(sampled_reads, "sampled_reads.fasta", "fasta")

# Run Kraken2 for taxonomic classification
subprocess.run(["kraken2", "--db", "kraken_db", "sampled_reads.fasta", 
                "--output", "kraken_out.txt", "--report", "kraken_report.txt"])
```

### Assembler Selection
- **Flye**:
  - Ideal for bacterial genomes
  - Excels with varied coverage and read lengths
  - Good repeat resolution
  - Efficient computation usage
  
- **Canu**:
  - Produces highly accurate assemblies
  - More computationally intensive
  - Works well with lower coverage
  - Good error correction
  
- **Raven/wtdbg2**:
  - Faster assembly with lower resource usage
  - Slightly less accurate
  - Good for quick preliminary assemblies
  
- **Specialized applications**:
  - Unicycler for hybrid Illumina/Nanopore data
  - NECAT for challenging genomes
  - Shasta for human-scale genomes

### Assembly Polishing Strategy
- **Long-read polishing**:
  - Medaka (Oxford Nanopore-specific polisher)
  - Racon (general consensus tool)
  - Multiple rounds may be beneficial
  
- **Short-read polishing (if available)**:
  - Pilon or NextPolish
  - Significantly reduces error rate
  - 2-3 rounds typically sufficient
  
- **Polishing evaluation**:
  - BUSCO scores to track improvement
  - Error rate estimation with mapping quality
  - Diminishing returns after 2-3 rounds

### Circular Genome Handling
For bacterial genomes:
- Check if assembly is circular (ends overlap)
- Rotate to start at a standard gene (e.g., dnaA or other origin)
- Remove duplicate sequence at ends
- Tools: Circlator or manual inspection

## Interpretation Considerations

### Assembly Quality Assessment
Evaluate your assembly using multiple metrics:
- **Contiguity**: N50, largest contig, number of contigs
- **Completeness**: BUSCO or CheckM scores
- **Accuracy**: Consensus accuracy, mapping quality
- **Structural correctness**: Alignment to reference genomes of related species

```python
# Example: Calculating basic assembly stats
from Bio import SeqIO

assembly = [rec for rec in SeqIO.parse("assembly.fasta", "fasta")]
lengths = sorted([len(rec) for rec in assembly], reverse=True)
total_length = sum(lengths)

n50 = 0
running_sum = 0
for length in lengths:
    running_sum += length
    if running_sum >= total_length * 0.5:
        n50 = length
        break

print(f"Assembly size: {total_length:,} bp")
print(f"Number of contigs: {len(lengths)}")
print(f"Largest contig: {lengths[0]:,} bp")
print(f"N50: {n50:,} bp")
```

### Common Assembly Challenges
- **Repeat regions**:
  - Check coverage drops in repeat regions
  - Look for collapse or expansion of repeats
  - Consider manual curation of known difficult regions
  
- **GC bias**:
  - Check for gaps in extreme GC content regions
  - May need coverage-aware assembly parameters
  
- **Contamination**:
  - Unexpected contig sizes or numbers
  - BLAST or taxonomic analysis of small contigs
  - CheckM for detecting mixed genomes

### Post-Assembly Analysis
After successful assembly:
- Gene prediction with Prokka
- Comparative genomics with related species
- Functional annotation
- Identification of genomic islands and horizontal gene transfer

## Technology-Specific Considerations
- **Error profile**:
  - ONT reads have higher indel than substitution error rates
  - Homopolymer regions are particularly challenging
  - Systematic errors in certain sequence contexts
  
- **Basecalling improvement**:
  - Re-basecalling with newer models can improve assembly
  - High-accuracy basecalling modes trade speed for accuracy
  - Consider GPU acceleration for basecalling
"""
        )
    ],
    references=[
        "Altschul SF, et al. (1990). Basic local alignment search tool. Journal of Molecular Biology.",
        "Katoh K, Standley DM. (2013). MAFFT multiple sequence alignment software version 7: improvements in performance and usability. Molecular Biology and Evolution.",
        "Li H. (2018). Minimap2: pairwise alignment for nucleotide sequences. Bioinformatics.",
        "Bailey TL, et al. (2009). MEME SUITE: tools for motif discovery and searching. Nucleic Acids Research.",
        "Nguyen LT, et al. (2015). IQ-TREE: a fast and effective stochastic algorithm for estimating maximum-likelihood phylogenies. Molecular Biology and Evolution.",
        "Kolmogorov M, et al. (2019). Assembly of long, error-prone reads using repeat graphs. Nature Biotechnology."
    ]
)

# Export the prompt for use in the package
if __name__ == "__main__":
    # Test the prompt with a sample query
    user_query = "How do I compare protein sequences from different bacterial species?"
    
    # Generate prompt
    prompt = sequence_analysis_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../sequence_analysis_prompt.json", "w") as f:
        f.write(sequence_analysis_prompt.to_json())

   # Load prompt template from JSON
    with open("../sequence_analysis_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt