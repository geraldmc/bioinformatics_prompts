"""Module containing the metagenomics research_area prompt template."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a metagenomics prompt template
metagenomics_prompt = BioinformaticsPrompt(
    research_area="Metagenomics",
    description=(
        "Metagenomics is the study of genetic material recovered directly from environmental samples. "
        "It enables the analysis of microbial communities without the need for culturing individual species. "
        "This field has expanded rapidly with advances in high-throughput sequencing technologies, "
        "allowing researchers to study complex microbiomes and their roles in environmental and health contexts."
    ),
    key_concepts=[
        "Microbial community profiling (taxonomic classification)",
        "Alpha and beta diversity analysis",
        "Shotgun metagenomics vs. amplicon sequencing (16S, ITS)",
        "Functional annotation and pathway analysis",
        "Metagenome assembly and binning",
        "Microbiome-host interactions",
        "Environmental metagenomics",
        "Human microbiome analysis"
    ],
    common_tools=[
        "QIIME/QIIME2 - Microbiome analysis pipeline",
        "MetaPhlAn - Profiling composition of microbial communities",
        "Kraken2/Bracken - Taxonomic classification of metagenomics reads",
        "DADA2 - Amplicon sequence variant inference",
        "SPAdes/MEGAHIT - Metagenomic assembly",
        "MetaBAT/MaxBin - Metagenomic binning",
        "MG-RAST - Metagenome annotation server",
        "HUMAnN - Functional profiling of microbial communities",
        "mothur - 16S rRNA gene analysis"
    ],
    common_file_formats=[
        {"name": "FASTQ", "description": "Raw sequencing reads with quality scores"},
        {"name": "FASTA", "description": "Nucleotide or protein sequences without quality scores"},
        {"name": "BIOM", "description": "Biological Observation Matrix for storing taxon/OTU tables"},
        {"name": "QIIME2 Artifact (.qza)", "description": "QIIME2-specific data format with provenance tracking"},
        {"name": "Phylogenetic trees (.nwk, .tre)", "description": "Newick format for phylogenetic trees"},
        {"name": "OTU/ASV tables", "description": "Tables of Operational Taxonomic Units or Amplicon Sequence Variants"}
    ],
    examples=[
        FewShotExample(
            query="How do I analyze 16S rRNA amplicon data from human gut microbiome samples?",
            context="The user has paired-end Illumina sequences targeting the V3-V4 region of the 16S rRNA gene from 50 human gut samples (25 healthy controls, 25 IBD patients).",
            response="""
To analyze 16S rRNA amplicon data from human gut microbiome samples, I recommend using QIIME2 with the following workflow:

1. Import your raw data into QIIME2 format:
   ```bash
   # For paired-end reads
   qiime tools import \
     --type 'SampleData[PairedEndSequencesWithQuality]' \
     --input-path manifest.tsv \
     --output-path paired-end-demux.qza \
     --input-format PairedEndFastqManifestPhred33V2
   ```

2. Generate sequence quality plots to determine trimming parameters:
   ```bash
   qiime demux summarize \
     --i-data paired-end-demux.qza \
     --o-visualization demux-summary.qzv
   ```

3. Denoise with DADA2 to obtain ASVs (Amplicon Sequence Variants):
   ```bash
   qiime dada2 denoise-paired \
     --i-demultiplexed-seqs paired-end-demux.qza \
     --p-trim-left-f 17 --p-trim-left-r 21 \
     --p-trunc-len-f 280 --p-trunc-len-r 230 \
     --o-table table.qza \
     --o-representative-sequences rep-seqs.qza \
     --o-denoising-stats denoising-stats.qza \
     --p-n-threads 0
   ```

4. Generate a phylogenetic tree for diversity analyses:
   ```bash
   qiime phylogeny align-to-tree-mafft-fasttree \
     --i-sequences rep-seqs.qza \
     --o-alignment aligned-rep-seqs.qza \
     --o-masked-alignment masked-aligned-rep-seqs.qza \
     --o-tree unrooted-tree.qza \
     --o-rooted-tree rooted-tree.qza
   ```

5. Assign taxonomy using a reference database (Silva or Greengenes):
   ```bash
   qiime feature-classifier classify-sklearn \
     --i-classifier silva-138-99-515-806-nb-classifier.qza \
     --i-reads rep-seqs.qza \
     --o-classification taxonomy.qza
   ```

6. Create taxonomy visualization:
   ```bash
   qiime metadata tabulate \
     --m-input-file taxonomy.qza \
     --o-visualization taxonomy.qzv
   
   qiime taxa barplot \
     --i-table table.qza \
     --i-taxonomy taxonomy.qza \
     --m-metadata-file sample-metadata.tsv \
     --o-visualization taxa-bar-plots.qzv
   ```

7. Perform alpha and beta diversity analyses:
   ```bash
   qiime diversity core-metrics-phylogenetic \
     --i-phylogeny rooted-tree.qza \
     --i-table table.qza \
     --p-sampling-depth 10000 \
     --m-metadata-file sample-metadata.tsv \
     --output-dir core-metrics-results
   ```

8. Test for differential abundance between IBD and healthy controls:
   ```bash
   qiime composition add-pseudocount \
     --i-table table.qza \
     --o-composition-table comp-table.qza
     
   qiime composition ancom \
     --i-table comp-table.qza \
     --m-metadata-file sample-metadata.tsv \
     --m-metadata-column disease_status \
     --o-visualization ancom-disease_status.qzv
   ```

9. For more detailed differential abundance analysis, use alternative methods:
   ```bash
   # Using the q2-aldex2 plugin
   qiime aldex2 aldex2 \
     --i-table table.qza \
     --m-metadata-file sample-metadata.tsv \
     --m-metadata-column disease_status \
     --o-visualization aldex2-disease_status.qzv
   ```

Key considerations:
- Adjust trimming parameters based on your quality profiles
- Choose a sampling depth that balances retained samples vs. sequence depth
- Consider batch effects if samples were processed in different runs
- Verify that your metadata file contains all necessary variables (treatment, age, gender, etc.)
- For human gut microbiome analysis, track dietary information and medication use when possible
"""
        ),
        FewShotExample(
            query="What's the best approach for taxonomic and functional profiling of shotgun metagenomic data?",
            context="The user has shotgun metagenomic data from soil samples and wants to identify both the microbial composition and metabolic potential of the community.",
            response="""
For taxonomic and functional profiling of shotgun metagenomic data from soil samples, I recommend this comprehensive workflow:

1. Quality control and preprocessing:
   ```bash
   # Trim adapters and low-quality sequences
   trimmomatic PE raw_reads_1.fastq.gz raw_reads_2.fastq.gz \
     clean_1.fastq.gz unpaired_1.fastq.gz \
     clean_2.fastq.gz unpaired_2.fastq.gz \
     ILLUMINACLIP:TruSeq3-PE.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:50
   
   # Remove host contamination if relevant
   bowtie2 -x host_genome -1 clean_1.fastq.gz -2 clean_2.fastq.gz --un-conc-gz filtered_%.fastq.gz
   ```

2. Taxonomic profiling with multiple tools for robustness:
   ```bash
   # Using Kraken2 + Bracken for taxonomic classification
   kraken2 --db kraken2_db --paired filtered_1.fastq.gz filtered_2.fastq.gz \
     --output kraken2_output.txt --report kraken2_report.txt
   
   bracken -d kraken2_db -i kraken2_report.txt -o bracken_output.txt -r 150 -l S
   
   # Using MetaPhlAn for strain-level profiling
   metaphlan filtered_1.fastq.gz,filtered_2.fastq.gz \
     --input_type fastq --bowtie2out metagenome.bowtie2.bz2 \
     --nproc 16 --output_file metaphlan_profile.txt
   ```

3. Functional profiling:
   ```bash
   # Using HUMAnN3 for metabolic pathway analysis
   humann --input filtered_1.fastq.gz --input-format fastq \
     --output humann_output --nucleotide-database chocophlan \
     --protein-database uniref90 --threads 16
   
   # Normalize and join tables if you have multiple samples
   humann_renorm_table --input humann_output/genefamilies.tsv \
     --output humann_output/genefamilies_relab.tsv --units relab
   
   humann_join_tables --input humann_output/ --output all_samples_genefamilies.tsv \
     --file_name genefamilies_relab
   
   humann_join_tables --input humann_output/ --output all_samples_pathabundance.tsv \
     --file_name pathabundance_relab
   ```

4. For soil metagenomes specifically, consider assembly and binning:
   ```bash
   # Assembly with MEGAHIT
   megahit -1 filtered_1.fastq.gz -2 filtered_2.fastq.gz -o assembly --min-contig-len 1000
   
   # Map reads back to assembly
   bowtie2-build assembly/final.contigs.fa assembly_index
   bowtie2 -x assembly_index -1 filtered_1.fastq.gz -2 filtered_2.fastq.gz | \
     samtools view -bS - > mapped.bam
   samtools sort mapped.bam -o sorted.bam
   samtools index sorted.bam
   
   # Binning with MetaBAT2
   jgi_summarize_bam_contig_depths --outputDepth depth.txt sorted.bam
   metabat2 -i assembly/final.contigs.fa -a depth.txt -o bins/bin
   ```

5. Functional annotation of assembled contigs:
   ```bash
   # Predict genes using Prodigal
   prodigal -i assembly/final.contigs.fa -a proteins.faa -d genes.fna -p meta
   
   # Annotate with eggNOG-mapper
   emapper.py -i proteins.faa --output eggnog_output -m diamond
   
   # Or use more soil-specific databases like KEGG or CAZy
   diamond blastp --db kegg.dmnd --query proteins.faa --out kegg_hits.tsv
   ```

6. Advanced soil-specific analysis:
   ```bash
   # Identify antimicrobial resistance genes
   abricate --db card assembly/final.contigs.fa > amr_genes.tab
   
   # Nitrogen cycle genes
   hmmsearch --tblout nitrogen_cycle.hits nitrogen_cycle_hmms.hmm proteins.faa
   ```

7. Visualize and integrate results:
   ```bash
   # Generate Krona charts for taxonomy
   ktImportTaxonomy -q 2 -t 3 kraken2_output.txt -o kraken2_taxonomy.html
   
   # Visualize functional pathways using Cytoscape or R
   Rscript -e "library(ggplot2); library(reshape2); data <- read.table('all_samples_pathabundance.tsv', header=TRUE, sep='\t'); ..."
   ```

Key considerations for soil metagenomes:
- Soil has extremely high microbial diversity; adjust parameters accordingly
- Be aware of extraction biases with soil samples
- Consider using specialized soil microbiome databases for better annotations
- Track soil properties (pH, moisture, nutrients) as metadata for correlations
- For ecological interpretation, focus on functional pathways relevant to soil processes (carbon cycling, nitrogen fixation)
"""
        ),
        FewShotExample(
            query="How can I recover Metagenome-Assembled Genomes (MAGs) from my environmental samples?",
            context="The user has deep shotgun metagenomic sequencing (~100Gb) from 5 hot spring samples and wants to recover high-quality microbial genomes.",
            response="""
Here's a comprehensive workflow for recovering high-quality Metagenome-Assembled Genomes (MAGs) from your hot spring samples:

1. Quality control and preprocessing:
   ```bash
   # For each sample
   fastp -i raw_reads_1.fastq.gz -I raw_reads_2.fastq.gz \
     -o clean_1.fastq.gz -O clean_2.fastq.gz \
     --detect_adapter_for_pe --correction --cut_right \
     --cut_window_size 4 --cut_mean_quality 20 --html fastp_report.html
   ```

2. Assemble each sample individually:
   ```bash
   # Using MEGAHIT for each sample
   for sample in sample1 sample2 sample3 sample4 sample5; do
     megahit -1 ${sample}_clean_1.fastq.gz -2 ${sample}_clean_2.fastq.gz \
       -o assembly_${sample} --min-contig-len 2000 \
       --k-min 27 --k-max 127 --k-step 10 --memory 0.8 --num-cpu-threads 32
   done
   ```

3. Map reads back to their respective assemblies:
   ```bash
   # For each sample
   for sample in sample1 sample2 sample3 sample4 sample5; do
     # Build index
     bowtie2-build assembly_${sample}/final.contigs.fa ${sample}_index
     
     # Map reads
     bowtie2 -x ${sample}_index -1 ${sample}_clean_1.fastq.gz -2 ${sample}_clean_2.fastq.gz | \
       samtools view -bS - > ${sample}_mapped.bam
     
     # Sort and index
     samtools sort ${sample}_mapped.bam -o ${sample}_sorted.bam
     samtools index ${sample}_sorted.bam
   done
   ```

4. Perform binning with multiple tools and co-assembly approach:
   ```bash
   # For individual samples first
   for sample in sample1 sample2 sample3 sample4 sample5; do
     # MetaBAT2
     jgi_summarize_bam_contig_depths --outputDepth ${sample}_depth.txt ${sample}_sorted.bam
     metabat2 -i assembly_${sample}/final.contigs.fa -a ${sample}_depth.txt -o bins_metabat/${sample}_bin
     
     # MaxBin2
     run_MaxBin.pl -contig assembly_${sample}/final.contigs.fa -out bins_maxbin/${sample}_bin \
       -reads1 ${sample}_clean_1.fastq.gz -reads2 ${sample}_clean_2.fastq.gz -thread 16
     
     # CONCOCT
     cut_up_fasta.py assembly_${sample}/final.contigs.fa -c 10000 -o 0 --merge_last -b ${sample}_contigs_10K.bed > ${sample}_contigs_10K.fa
     concoct_coverage_table.py ${sample}_contigs_10K.bed ${sample}_sorted.bam > ${sample}_coverage_table.tsv
     concoct --composition_file ${sample}_contigs_10K.fa --coverage_file ${sample}_coverage_table.tsv -b concoct_output/${sample}
   done
   
   # Optional: Also try co-assembly for closely related samples
   cat sample1_clean_1.fastq.gz sample2_clean_1.fastq.gz > coassembly_R1.fastq.gz
   cat sample1_clean_2.fastq.gz sample2_clean_2.fastq.gz > coassembly_R2.fastq.gz
   megahit -1 coassembly_R1.fastq.gz -2 coassembly_R2.fastq.gz -o assembly_coassembly --min-contig-len 2000
   ```

5. Use DASTool to integrate and optimize bins from different binners:
   ```bash
   for sample in sample1 sample2 sample3 sample4 sample5; do
     # Convert binning results to DASTool format
     Fasta_to_Contig2Bin.sh -i bins_metabat/${sample}_bin -e fa > ${sample}_metabat_contigs2bin.tsv
     Fasta_to_Contig2Bin.sh -i bins_maxbin/${sample}_bin -e fasta > ${sample}_maxbin_contigs2bin.tsv
     perl scripts/concoct_bins_to_dastool.pl concoct_output/${sample}_clustering_gt1000.csv > ${sample}_concoct_contigs2bin.tsv
     
     # Run DASTool
     DAS_Tool -i ${sample}_metabat_contigs2bin.tsv,${sample}_maxbin_contigs2bin.tsv,${sample}_concoct_contigs2bin.tsv \
       -c assembly_${sample}/final.contigs.fa -o dastool_output/${sample} \
       --write_bins 1 --write_unbinned 1 --threads 16
   done
   ```

6. Assess bin quality with CheckM and GTDB-Tk:
   ```bash
   # CheckM for completeness and contamination
   checkm lineage_wf dastool_output/sample1_DASTool_bins/ checkm_output/sample1 -x fa -t 16
   
   # GTDB-Tk for taxonomic classification
   gtdbtk classify_wf --genome_dir dastool_output/sample1_DASTool_bins/ \
     --out_dir gtdbtk_output/sample1 --extension fa --cpus 16
   ```

7. Filter for high-quality MAGs:
   ```bash
   # Create a list of high-quality MAGs (>90% completeness, <5% contamination)
   checkm qa checkm_output/sample1/lineage.ms checkm_output/sample1 -o 2 -f high_quality_mags.tsv --tab_table
   
   # Filter based on quality metrics
   awk -F'\t' '($13 >= 90) && ($14 <= 5) {print $1}' high_quality_mags.tsv > high_quality_mags_list.txt
   ```

8. Improve bin quality with post-binning refinement (optional):
   ```bash
   # Use MetaWRAP's bin refinement module for problematic bins
   metawrap bin_refinement -o bin_refinement \
     -t 16 \
     -A bins_metabat/${sample}_bin \
     -B bins_maxbin/${sample}_bin \
     -C concoct_output/${sample}_bins \
     -c 70 -x 10
   ```

9. Annotate MAGs for functional analysis:
   ```bash
   # Predict genes
   for mag in high_quality_mags/*.fa; do
     basename=$(basename $mag .fa)
     prodigal -i $mag -a ${basename}_proteins.faa -d ${basename}_genes.fna -p single
   done
   
   # Annotate with KEGG
   for protein in high_quality_mags/*_proteins.faa; do
     basename=$(basename $protein _proteins.faa)
     kofamscan -o ${basename}_kofam.txt -p /path/to/profiles -k /path/to/ko_list $protein
   done
   ```

Key considerations for hot spring MAGs:
- Hot springs often harbor unique thermophilic microbes; consider reference databases that include thermophiles
- Be aware that some hot spring microbes have unusual genomic properties (e.g., high GC content)
- co-assembly may work well for similar hot spring samples but could complicate binning if communities differ significantly
- Use stricter quality thresholds for publication-quality MAGs (>95% completion, <5% contamination)
- Consider dereplication if you recover similar MAGs across samples (using dRep)
- For thermophilic environments, specialized analyses of heat-stable enzymes might be valuable
"""
        )
    ],
    references=[
        "Quince C, et al. (2017). Shotgun metagenomics, from sampling to analysis. Nature Biotechnology.",
        "Bolyen E, et al. (2019). Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2. Nature Biotechnology.",
        "Franzosa EA, et al. (2018). Species-level functional profiling of metagenomes and metatranscriptomes. Nature Methods.",
        "Kang DD, et al. (2019). MetaBAT 2: an adaptive binning algorithm for robust and efficient genome reconstruction from metagenome assemblies. PeerJ."
    ]
)

# Export the prompt for use in the package
if __name__ == "__main__":
    # Test the prompt with a sample query
    user_query = "How can I compare the taxonomic composition between different environmental samples?"
    
    # Generate prompt
    prompt = metagenomics_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../metagenomics_prompt.json", "w") as f:
        f.write(metagenomics_prompt.to_json())

   # Load prompt template from JSON
    with open("../metagenomics_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt