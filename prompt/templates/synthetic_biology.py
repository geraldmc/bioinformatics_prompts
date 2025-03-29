import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a synthetic biology prompt template
synthetic_biology_prompt = BioinformaticsPrompt(
    discipline="Synthetic Biology",
    description=(
        "Synthetic biology integrates biology, engineering, and computational approaches to design, "
        "model, and construct novel biological systems. This field applies engineering principles to "
        "biology, developing standardized genetic parts and circuits with predictable functions. "
        "Bioinformatics in synthetic biology supports genetic circuit design, protein engineering, "
        "metabolic pathway optimization, and whole-genome design, enabling both bottom-up construction "
        "of synthetic systems and top-down redesign of existing organisms for applications in "
        "medicine, biomanufacturing, agriculture, and environmental remediation."
    ),
    key_concepts=[
        "Genetic circuit design and modeling",
        "Parts-based genetic engineering (BioBricks, MoClo, Golden Gate)",
        "Metabolic pathway engineering and flux optimization",
        "Protein design and directed evolution",
        "Genome refactoring and minimization",
        "Biosafety and biocontainment strategies",
        "Design-Build-Test-Learn cycle optimization",
        "Computer-aided design (CAD) for synthetic biology"
    ],
    common_tools=[
        "SynBioHub - Repository for synthetic biology parts and designs",
        "Benchling - Collaborative life sciences R&D platform",
        "iBioSim/Cello - Genetic circuit design and simulation",
        "Geneious/SnapGene - DNA design and visualization",
        "COBRA/GECKO - Genome-scale metabolic modeling",
        "j5/DIVA/SAVI - DNA assembly design",
        "BOOST/Codon Juggler - Codon optimization",
        "Rosetta/AlphaFold - Protein design and structure prediction"
    ],
    common_file_formats=[
        {"name": "SBOL/SBOL Visual", "description": "Synthetic Biology Open Language for genetic designs"},
        {"name": "GenBank/FASTA", "description": "DNA sequence and annotation formats"},
        {"name": "SBML", "description": "Systems Biology Markup Language for models"},
        {"name": "SBGN", "description": "Systems Biology Graphical Notation for pathways"},
        {"name": "SBOLv", "description": "Visual standard for synthetic biology diagrams"},
        {"name": "PDB/mmCIF", "description": "Protein structure data formats"},
        {"name": "CSV/TSV", "description": "Tabular format for experimental data and parameters"}
    ],
    examples=[
        FewShotExample(
            query="What are the key design considerations for engineering a metabolic pathway for bioproduction in bacteria?",
            context="The user wants to engineer a bacterial strain to produce a high-value chemical and needs guidance on the design approach and decision points.",
            response="""
# Metabolic Pathway Engineering for Bioproduction

## Conceptual Framework

Engineering metabolic pathways for bioproduction involves multiple interconnected design considerations:

1. **Production objective alignment**
2. **Host selection and compatibility**
3. **Pathway design and optimization**
4. **Regulatory circuit integration**
5. **Scaling and robustness planning**

Success requires iterative refinement through the Design-Build-Test-Learn cycle.

## Decision Framework

### Host Selection Decisions
- **Chassis organism selection**:
  - **E. coli**: Rapid growth, genetic tractability, extensive toolkits
  - **B. subtilis**: Secretion capacity, GRAS status, stress tolerance
  - **S. cerevisiae**: Eukaryotic processing, tolerance to harsh conditions
  - **Non-model organisms**: Specialized metabolic capabilities, unique tolerance profiles
  
- **Host strain background**:
  - Laboratory vs. industrial strains
  - Genomically reduced vs. wild-type strains
  - Auxotrophic vs. prototrophic requirements
  - Existing engineered capabilities (e.g., RNA polymerase modifications)

- **Host-pathway compatibility factors**:
  - Cofactor availability (NAD(P)H, ATP balancing)
  - Redox state management
  - Toxic intermediate tolerance
  - Growth-production balancing mechanisms

### Pathway Design Considerations
- **Pathway source strategy**:
  - Native vs. heterologous pathway components
  - Orthogonal elements to minimize crosstalk
  - Synthetic vs. natural enzyme variants
  - Multi-organism inspired hybrid pathways

- **Enzyme selection criteria**:
  - Kinetic parameters (Km, kcat) alignment with production goals
  - Expression compatibility in host (codon usage, folding)
  - Allosteric regulation and feedback sensitivity
  - Cofactor specificity (NADH vs. NADPH preference)

- **Pathway topology decisions**:
  - Linear vs. branched architecture
  - Bottleneck identification and mitigation
  - Side reaction minimization strategies
  - Compartmentalization opportunities

### Expression Control Strategy
- **Transcriptional regulation approach**:
  - Constitutive vs. inducible promoters
  - Promoter strength tuning strategy
  - Dynamic regulatory systems (metabolite-responsive)
  - Growth phase-dependent expression

- **Translation optimization**:
  - RBS strength selection and balancing
  - Codon optimization strategy
  - mRNA stability considerations
  - Protein degradation tag usage

- **Enzyme ratio balancing**:
  - Operon vs. individual promoter architecture
  - Ribosome binding site libraries
  - Copy number and plasmid selection
  - Chromosomal integration vs. plasmid-based expression

### Metabolic Integration Planning
- **Competing pathway management**:
  - Gene knockout prioritization
  - Conditional repression strategies
  - Dynamic flux redirection approaches
  - Essential pathway balancing

- **Precursor supply enhancement**:
  - Central metabolism engineering strategy
  - Overflow metabolism management
  - Auxiliary pathway upregulation
  - Transport system optimization

- **Cofactor and energy balance**:
  - Redox balance (NAD+/NADH, NADP+/NADPH)
  - ATP generation/consumption ratio management
  - Electron transport chain modifications
  - Futile cycle elimination

### Implementation and Testing Strategy
- **Construction approach selection**:
  - Modular cloning system (MoClo, Golden Gate, Gibson)
  - Multiplex genome editing (CRISPR-Cas, MAGE)
  - Assembly verification strategy
  - Strain archiving protocol

- **Screening and selection design**:
  - High-throughput screening approach
  - Biosensor implementation
  - Growth-production coupling strategy
  - Analytical method selection

- **Scale-up consideration timeline**:
  - Early-stage scale-up indicators
  - Process constraints identification
  - Robustness to industrial conditions
  - Genetic stability assessment

## Implementation Considerations

- **Start with computational modeling**: Utilize genome-scale metabolic models to predict pathway behavior
- **Design modular components**: Create exchangeable parts for rapid iteration
- **Implement dynamic regulation**: Consider dynamic control systems responsive to metabolic state
- **Plan for robustness**: Design strategies to minimize mutation and metabolic burden
- **Consider separating growth and production phases**: Two-stage fermentation strategies

## Useful Resources

- [iGEM Registry of Standard Biological Parts](https://parts.igem.org/)
- [KEGG Pathway Database](https://www.genome.jp/kegg/pathway.html)
- [EcoCyc E. coli Database](https://ecocyc.org/)
- [Synthetic Biology Open Language (SBOL)](https://sbolstandard.org/)
- [COBRA Toolbox for metabolic modeling](https://opencobra.github.io/cobratoolbox/)
"""
        ),
        FewShotExample(
            query="How should I approach designing genetic circuits with predictable behavior for biosensing applications?",
            context="The user wants to design a genetic circuit that can detect and report the presence of specific environmental compounds.",
            response="""
# Genetic Circuit Design for Biosensing Applications

## Conceptual Framework

Designing genetic biosensors requires balancing:

1. **Input sensing**: Specific and sensitive detection
2. **Signal processing**: Appropriate computational operations
3. **Output generation**: Measurable reporter expression
4. **System integration**: Function within cellular context
5. **Performance optimization**: Tuning response characteristics

Effective design navigates tradeoffs between sensitivity, specificity, and robustness.

## Decision Framework

### Sensing Module Design
- **Sensor mechanism selection**:
  - **Transcription factor-based**: 
    - One-component (e.g., TetR, LacI) vs. two-component systems
    - Advantages: Well-characterized, tunable
    - Limitations: Limited analyte range, potential crosstalk
  
  - **RNA-based**:
    - Riboswitches vs. toehold switches vs. aptamer-based
    - Advantages: Programmable, compact, faster response
    - Limitations: Often lower dynamic range, context sensitivity
  
  - **CRISPR-based sensing**:
    - dCas9/Cas12/Cas13 approaches
    - Advantages: Programmability, multiplexing potential
    - Limitations: System complexity, resource requirements

- **Input-sensing characteristics**:
  - Detection threshold requirements
  - Dynamic range needs
  - Response time constraints
  - Specificity vs. cross-reactivity tolerance

- **Sensor selection criteria**:
  - Natural vs. engineered sensor elements
  - Binding affinity (Kd) matching with detection needs
  - Host compatibility considerations
  - Off-target interaction potential

### Signal Processing Architecture
- **Circuit topology selection**:
  - **Digital logic implementation**:
    - Boolean operations (AND, OR, NOT, NOR)
    - Memory elements (toggle switches, flip-flops)
    - Edge detection or pulse generation
  
  - **Analog computing elements**:
    - Positive/negative feedback loops
    - Feed-forward loops
    - Proportional-integral control motifs
  
  - **Signal processing requirements**:
    - Noise filtering needs
    - Amplification requirements
    - Temporal considerations (persistence, adaptation)
    - Multi-input integration strategy

- **Component selection and design**:
  - Orthogonality requirements between components
  - Parameter matching for connected modules
  - Resource competition mitigation
  - Modularity vs. optimization tradeoffs

### Output Module Considerations
- **Reporter selection criteria**:
  - **Fluorescent proteins**:
    - Maturation time, brightness, stability
    - Spectral characteristics and equipment compatibility
    - Single reporters vs. FRET pairs
  
  - **Enzymatic reporters**:
    - Amplification potential
    - Substrate requirements and toxicity
    - Quantification approach
  
  - **Growth/survival coupling**:
    - Essential gene regulation
    - Toxin-antitoxin systems
    - Selective advantage mechanisms

- **Output characteristics design**:
  - Continuous vs. threshold response
  - Reversibility requirements
  - Signal persistence needs
  - Multiplexing considerations

### Host and Context Compatibility
- **Cellular resource allocation**:
  - Transcription/translation burden assessment
  - Resource competition mitigation strategies
  - Growth impact minimization
  - Metabolic load balancing

- **Host background consideration**:
  - Endogenous regulatory interference
  - Physiological state dependencies
  - Growth phase impact on performance
  - Environmental sensitivity

- **Implementation format**:
  - Plasmid vs. chromosomal integration
  - Copy number selection
  - Stability considerations
  - Mobility requirements

### Characterization and Optimization Strategy
- **Characterization approach**:
  - Dose-response curve measurement
  - Time-course dynamics assessment
  - Cross-reactivity testing
  - Robustness evaluation across conditions

- **Parameter tuning strategies**:
  - RBS/promoter strength libraries
  - Protein degradation tag tuning
  - Operator site affinity modification
  - DNA/RNA stability modulation

- **Robustness engineering**:
  - Insulation from host variation
  - Adaptation to environmental fluctuations
  - Long-term evolutionary stability
  - Fail-safe mechanism implementation

## Implementation Considerations

- **Modular testing approach**: Characterize individual components before assembly
- **Context effects anticipation**: Test parts in final configuration and host
- **Standards adherence**: Use standard assembly methods for reproducibility
- **Forward design iteration**: Start simple, add complexity incrementally
- **Model-guided design**: Use computational models to predict behavior

## Useful Resources

- [SynBioHub](https://synbiohub.org/) - Repository of standard biological parts
- [Cello](http://www.cellocad.org/) - Genetic circuit design automation
- [iBioSim](https://www.async.ece.utah.edu/ibiosim/) - Circuit modeling and analysis
- [Benchling](https://www.benchling.com/) - Collaborative design platform
- [SBOL](https://sbolstandard.org/) - Standard for representing genetic designs
"""
        )
    ],
    references=[
        "Appleton E, et al. (2022). Design automation for synthetic biology. Nature Reviews Methods Primers, 2(1), 24.",
        "Reis AC, et al. (2023). Biofoundries enable automated design-build-test workflow for synthetic biology. Nature Biotechnology, 41(3), 334-347.",
        "Grozinger L, et al. (2022). Pathways to cellular supremacy in biocomputing. Nature Communications, 13(1), 1-11.",
        "Wang F, et al. (2021). Genome-scale metabolic network reconstruction and in silico analyses of the metabolically versatile bacterium Pseudomonas aeruginosa PAO1. Molecular Omics, 17(5), 767-779.",
        "Ko YS, et al. (2024). Development of a new ensemble model to predict the metabolic design of complex nonlinear pathways. Metabolic Engineering, 78, 46-58."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "How do I design a genetic circuit for biosensing environmental toxins?"
    
    # Generate prompt
    prompt = synthetic_biology_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../synthetic_biology_prompt.json", "w") as f:
        f.write(synthetic_biology_prompt.to_json())

    # Load prompt template from JSON
    with open("../synthetic_biology_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt