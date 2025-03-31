import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a genomics prompt template
genomics_prompt = BioinformaticsPrompt(
    research_area="Genomics",
    description=(
        "Genomics is the study of the complete set of DNA (including all genes) in an organism. "
        "It involves the sequencing, assembly, annotation, and analysis of genomes to understand "
        "genetic variation, gene function, and evolutionary relationships."
    ),
    key_concepts=[
        "DNA sequencing technologies (short-read, long-read)",
        "Genome assembly (de novo, reference-guided)",
        "Variant calling (SNPs, indels, structural variants)",
        "Gene prediction and annotation",
        "Comparative genomics",
        "Functional genomics",
        "Population genomics"
    ],
    common_tools=[
        "BWA/Bowtie2 - Read alignment",
        "GATK - Variant calling and analysis",
        "SPAdes/MEGAHIT - De novo assembly",
        "BLAST - Sequence similarity search",
        "IGV - Genome visualization",
        "Prokka/MAKER - Genome annotation",
        "samtools/bcftools - SAM/BAM/VCF manipulation"
    ],
    common_file_formats=[
        {"name": "FASTQ", "description": "Raw sequencing reads with quality scores"},
        {"name": "FASTA", "description": "Nucleotide or protein sequences without quality scores"},
        {"name": "SAM/BAM", "description": "Sequence alignment data"},
        {"name": "VCF", "description": "Variant call format for SNPs and other variants"},
        {"name": "GFF/GTF", "description": "Gene annotation format"},
        {"name": "BED", "description": "Browser extensible data format for genomic intervals"}
    ],
    examples=[
        FewShotExample(
            query="How do I identify SNPs in my bacterial genome compared to a reference?",
            context="The user has Illumina paired-end reads from a bacterial isolate and wants to identify single nucleotide polymorphisms.",
            response="""
To identify SNPs in your bacterial genome compared to a reference, follow this workflow:

1. Quality control your raw reads:
   ```bash
   fastqc -o qc_results raw_reads_R1.fastq.gz raw_reads_R2.fastq.gz
   ```

2. Trim low-quality bases and adapters:
   ```bash
   trimmomatic PE raw_reads_R1.fastq.gz raw_reads_R2.fastq.gz \
     trimmed_R1.fastq.gz unpaired_R1.fastq.gz \
     trimmed_R2.fastq.gz unpaired_R2.fastq.gz \
     ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36
   ```

3. Align reads to the reference genome:
   ```bash
   bwa index reference.fasta
   bwa mem reference.fasta trimmed_R1.fastq.gz trimmed_R2.fastq.gz | samtools view -bS - > aligned.bam
   ```

4. Sort and index BAM file:
   ```bash
   samtools sort aligned.bam -o aligned.sorted.bam
   samtools index aligned.sorted.bam
   ```

5. Call variants:
   ```bash
   bcftools mpileup -f reference.fasta aligned.sorted.bam | bcftools call -mv -Ob -o variants.bcf
   bcftools view variants.bcf > variants.vcf
   ```

6. Filter variants:
   ```bash
   bcftools filter -i 'QUAL>20 && DP>10' variants.vcf > filtered_variants.vcf
   ```

7. Annotate variants (optional):
   ```bash
   snpEff -v -csvStats snp_stats.csv reference_db filtered_variants.vcf > annotated_variants.vcf
   ```

Key considerations:
- Ensure your reference genome is closely related to your isolate
- Adjust filtering parameters based on your sequencing depth
- Check that your SNPs have good coverage and quality scores
"""
        ),
        FewShotExample(
            query="What's the best approach for de novo assembly of a bacterial genome from Oxford Nanopore data?",
            context="The user has Oxford Nanopore long reads (~20x coverage) from a novel bacterial strain.",
            response="""
For de novo assembly of a bacterial genome from Oxford Nanopore data, I recommend this workflow:

1. Quality control and filtering:
   ```bash
   NanoPlot --fastq reads.fastq --outdir qc_results
   NanoFilt -q 10 -l 1000 reads.fastq > filtered_reads.fastq
   ```

2. For pure long-read assembly, use Flye:
   ```bash
   flye --nano-raw filtered_reads.fastq --out-dir assembly --genome-size 5m --threads 16
   ```

3. Polish the assembly to reduce error rate (multiple rounds recommended):
   ```bash
   # Map reads to the assembly
   minimap2 -ax map-ont assembly/assembly.fasta filtered_reads.fastq | samtools sort -o aligned.bam
   samtools index aligned.bam
   
   # Polish with Racon (1-2 rounds)
   racon filtered_reads.fastq aligned.paf assembly/assembly.fasta > polished.fasta
   
   # Further polish with Medaka
   medaka_consensus -i filtered_reads.fastq -d polished.fasta -o medaka_polished -t 16 -m r941_min_high_g360
   ```

4. Evaluate assembly quality:
   ```bash
   quast.py medaka_polished/consensus.fasta -o assembly_qc
   ```

5. Check for completeness using single-copy orthologs:
   ```bash
   busco -i medaka_polished/consensus.fasta -o busco_results -m genome -l bacteria_odb10
   ```

Key considerations:
- With only 20x coverage, you may see higher error rates. Consider additional polishing or hybrid assembly
- For higher quality, incorporate Illumina reads if available using tools like Pilon
- Long-read assembly is generally better for resolving repeat regions in bacterial genomes
- Check for potential contamination using tools like CheckM
"""
        )
    ],
    references=[
        "Sedlazeck FJ, et al. (2018). Accurate detection of complex structural variations using single-molecule sequencing. Nature Methods.",
        "Bush SJ, et al. (2020). Best practice in the application of bioinformatics to microbial genome annotation. Frontiers in Microbiology."
    ]
)

# FewShotExample usage
if __name__ == "__main__":
    # FewShotExample user query
    user_query = "I have Illumina paired-end reads from a bacterial sample. How can I assemble and annotate the genome?"
    
    # Generate prompt
    prompt = genomics_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../genomics_prompt.json", "w") as f:
        f.write(genomics_prompt.to_json())
    
    # Load prompt template from JSON
    with open("../genomics_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt