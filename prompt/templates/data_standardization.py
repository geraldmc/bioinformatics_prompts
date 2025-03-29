import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a data standardization prompt template
data_standardization_prompt = BioinformaticsPrompt(
    discipline="Bioinformatics Data Standardization and Interoperability",
    description=(
        "Data standardization and interoperability in bioinformatics enable effective sharing, integration, "
        "and reuse of biological data across research communities. As high-throughput technologies generate "
        "increasingly diverse and voluminous datasets, adherence to community standards and FAIR principles "
        "(Findable, Accessible, Interoperable, Reusable) becomes critical for collaborative research, "
        "reproducibility, and knowledge discovery. Standardization spans data formats, metadata schemas, "
        "vocabularies, and exchange protocols, creating an ecosystem where datasets can seamlessly flow "
        "between repositories, analysis platforms, and visualization tools."
    ),
    key_concepts=[
        "FAIR data principles and implementation strategies",
        "Metadata standards and minimal information requirements",
        "Ontologies and controlled vocabularies in bioinformatics",
        "Data validation and quality assessment frameworks",
        "Repository selection and submission workflows",
        "Data licensing and access control mechanisms",
        "Persistent identifiers and data citation practices",
        "Cross-domain data integration approaches"
    ],
    common_tools=[
        "ISA framework - Investigation/Study/Assay metadata tracking",
        "CEDAR - Center for Expanded Data Annotation and Retrieval",
        "FAIRshake/FAIRsharing - FAIR assessment and resource discovery",
        "BioSchemas - Structure data markup for life sciences",
        "OBO Foundry - Open Biological and Biomedical Ontologies",
        "GA4GH standards - Genomic data exchange specifications",
        "RDF/OWL/SPARQL - Semantic web technologies for linked data"
    ],
    common_file_formats=[
        {"name": "ISA-Tab/ISA-JSON", "description": "Structured metadata for omics experiments"},
        {"name": "MAGE-TAB", "description": "Microarray and transcriptomics experiment metadata"},
        {"name": "CRAM/BAM", "description": "Compressed sequence alignment formats"},
        {"name": "VCF/BCF", "description": "Variant Call Format and binary equivalent"},
        {"name": "CWL/WDL", "description": "Workflow definition languages for reproducible analysis"},
        {"name": "OWL/OBO", "description": "Ontology specification formats"},
        {"name": "HDF5/Zarr", "description": "Hierarchical data formats for multidimensional arrays"}
    ],
    examples=[
        FewShotExample(
            query="How should I approach making my genomics dataset FAIR-compliant for submission to a public repository?",
            context="The user has completed a large-scale genomics study and wants to ensure their data follows FAIR principles before repository submission.",
            response="""
# Making Genomics Datasets FAIR-Compliant

## Conceptual Framework

The FAIR principles provide a framework for optimizing data reusability through:

1. **Findability**: Unique identifiers, rich metadata, indexing
2. **Accessibility**: Retrievable data with clear access protocols
3. **Interoperability**: Using standard formats and vocabularies
4. **Reusability**: Clear usage licenses and provenance information

Implementing FAIR for genomics involves strategic decisions across the data lifecycle.

## Decision Framework

### Pre-Submission Planning
- **Repository selection criteria**:
  - Domain appropriateness (e.g., SRA for raw sequences, GEO for expression)
  - Long-term sustainability and funding model
  - Integration with other resources
  - Support for relevant identifiers (DOIs, ORCIDs)
  - Community recognition and usage
  
- **Identifier strategy**:
  - Sample identifiers: BioSample vs. internal IDs
  - Study/project: BioProject vs. repository-specific
  - File-level: UUID generation strategy
  - Supporting persistent identifiers (ORCIDs for contributors)

- **License selection**:
  - Open vs. controlled access considerations
  - Creative Commons license selection (CC0, CC-BY)
  - Data use agreements for sensitive data
  - Embargo periods and timing

### Metadata Enrichment Decisions
- **Metadata standard selection**:
  - Generic (Dublin Core, DataCite) vs. domain-specific (MIxS, MINSEQE)
  - Single vs. multiple complementary standards
  - Core/extended structure and completeness levels
  
- **Controlled vocabulary usage**:
  - OBO Foundry ontologies (OBI, EFO, SO, etc.)
  - Repository-required terminologies
  - Custom term requirements
  - Versioning considerations for ontologies
  
- **Experimental context documentation**:
  - Sample collection and processing details
  - Technical protocol documentation level
  - Quality control metrics to include
  - Batch information and potential confounders

### File Format and Organization
- **Format standardization**:
  - Raw data: FASTQ with quality encoding specification
  - Aligned data: BAM vs. CRAM tradeoffs
  - Derived data: Standard vs. specialized formats
  
- **Directory structure design**:
  - Hierarchical vs. flat organization
  - Naming conventions and consistency
  - README and manifest file design
  - Version control approach
  
- **Validation requirements**:
  - Format validation tools (e.g., ValidateFASTQ, Picard ValidateSamFile)
  - Metadata compliance checking
  - Consistency across related files
  - Pre-submission quality assessment

### Data Processing Documentation
- **Method transparency approach**:
  - Workflow language selection (CWL, WDL, Nextflow)
  - Software version documentation strategy
  - Parameter reporting completeness
  - Container vs. environment specifications
  
- **Compute environment preservation**:
  - Container creation (Docker, Singularity)
  - Environment capture (conda, pip freeze)
  - Hardware specification documentation
  - Runtime parameter logging

## Implementation Considerations

- **Start with repository guidelines**: Each repository has specific requirements
- **Layer standards appropriately**: Core standards plus domain-specific extensions
- **Automate metadata collection**: Integrate into analysis workflows
- **Validate early and often**: Check compliance throughout the research process
- **Consider machine-readability**: Balance human and machine-readable formats

## Useful Resources

- [FAIRsharing](https://fairsharing.org/) - Standards, databases, and policies
- [GA4GH FAIR Cookbook](https://faircookbook.elixir-europe.org/) - Implementation recipes 
- [EBI Submission Portals](https://www.ebi.ac.uk/submission/) - Unified submission interfaces
- [BioSchemas Profiles](https://bioschemas.org/profiles) - Markup specifications
- [CEDAR Metadata Templates](https://metadatacenter.org/) - Standardized metadata collection
"""
        ),
        FewShotExample(
            query="What are the key decisions for integrating multiple biomedical datasets that use different ontologies and standards?",
            context="The user wants to integrate clinical, genomic, and proteomics data from different sources that use varying standards and ontologies.",
            response="""
# Cross-Domain Biomedical Data Integration Strategies

## Conceptual Framework

Integrating heterogeneous biomedical datasets requires addressing three key challenges:

1. **Semantic harmonization**: Reconciling different terminologies and ontologies
2. **Technical standardization**: Handling diverse data formats and structures
3. **Knowledge representation**: Creating a unified data model that preserves context

A successful integration strategy must balance standardization with flexibility and scalability.

## Decision Framework

### Integration Approach Selection
- **Integration paradigm**:
  - **Centralized warehouse**: Transform all data to common model
    - Advantages: Consistent querying, optimized performance
    - Challenges: Transformation loss, update complexity
  
  - **Federated/distributed**: Query data in original sources
    - Advantages: Data remains at source, reduced transformation
    - Challenges: Query performance, source availability
  
  - **Knowledge graph**: Connected semantic representation
    - Advantages: Flexible schema, relationship focus
    - Challenges: Complex querying, infrastructure needs
  
  - **Hybrid approaches**: Combined strategies
    - Advantages: Balances tradeoffs between approaches
    - Challenges: Increased architectural complexity

### Semantic Mapping Strategy
- **Ontology alignment approach**:
  - Direct mapping between source ontologies
  - Mapping to upper-level ontology (BFO, UMLS)
  - Custom application ontology development
  - Terminology mapping services (e.g., OLS, ZOOMA)
  
- **Mapping complexity handling**:
  - One-to-one vs. one-to-many mappings
  - Handling partial concept matches
  - Managing hierarchical relationship differences
  - Temporal ontology version management
  
- **Mapping implementation**:
  - Manual expert curation vs. automated suggestions
  - Mapping format (SSSOM, SKOS, OWL)
  - Validation and quality control processes
  - Mapping provenance and versioning

### Technical Implementation Decisions
- **Data model selection**:
  - Relational vs. NoSQL approaches
  - Graph database applicability
  - Resource Description Framework (RDF) consideration
  - Hybrid storage architectures
  
- **Common data model adoption**:
  - Domain-specific models (OMOP, i2b2, tranSMART)
  - Generic models with extensions
  - Custom model development
  - Multiple coordinated models
  
- **Interface and access layer**:
  - API design and standardization (REST, GraphQL)
  - Query language selection (SQL, SPARQL, Cypher)
  - Authentication and authorization mechanisms
  - Cross-dataset query optimization

### Data Harmonization Process
- **Preprocessing standardization**:
  - Units of measurement normalization
  - Missing data handling strategy
  - Outlier identification approach
  - Data quality threshold setting
  
- **Entity resolution strategy**:
  - Patient/sample matching methods
  - Confidence scoring for entity links
  - Handling conflicting identifiers
  - Temporal alignment of measurements
  
- **Provenance tracking**:
  - Origin data source documentation
  - Transformation step recording
  - Quality/confidence metrics preservation
  - Update and versioning management

### Validation and Quality Assessment
- **Validation approach selection**:
  - Technical validation (format, syntax)
  - Semantic validation (meaning preservation)
  - Statistical validation (distribution comparison)
  - Expert validation (domain knowledge checks)
  
- **Quality metric development**:
  - Completeness assessment
  - Consistency checking
  - Accuracy validation
  - Timeliness evaluation

## Implementation Considerations

- **Start with clear use cases**: Define specific queries the integration should support
- **Prioritize critical data elements**: Focus on core concepts first, then expand
- **Implement iterative mapping**: Build and validate mappings incrementally
- **Consider long-term maintenance**: Design for updates to source data and ontologies
- **Document assumptions**: Explicitly record all mapping decisions and rationales

## Useful Resources

- [OHDSI OMOP Common Data Model](https://www.ohdsi.org/data-standardization/)
- [Biomedical Data Translator](https://ncats.nih.gov/translator) - Knowledge graph approach
- [Human Cell Atlas](https://data.humancellatlas.org/) - Data coordination platform
- [EMBL-EBI Ontology Xref Service](https://www.ebi.ac.uk/spot/oxo/) - Ontology mapping
- [NIH Bridge2AI Standards](https://bridge2ai.org/standards/) - Emerging standards
"""
        )
    ],
    references=[
        "Wilkinson MD, et al. (2023). FAIR principles for data stewardship four years on. Scientific Data, 10(1), 224.",
        "Gonzalez-Beltran A, Rocca-Serra P (2022). FAIRsharing, a platform to standardize minimum metadata requirements. Scientific Data, 9(1), 592.",
        "Barsnes H, Vaudel M (2022). Standardization in proteomics: The Human Proteome Organization's guidelines for mass spectrometry data. Journal of Proteome Research, 21(8), 1897-1900.",
        "Luo Y, et al. (2024). Harmonizing healthcare data standards with FHIR and OMOP for interoperable health information exchange. NPJ Digital Medicine, 7(1), 57.",
        "Queralt-Rosinach N, et al. (2021). Knowledge graphs and wikidata subsetting for rare disease cohort analytics. Scientific Data, 8(1), 294."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "What standards should I follow when preparing my multi-omics dataset for publication?"
    
    # Generate prompt
    prompt = data_standardization_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../data_standardization_prompt.json", "w") as f:
        f.write(data_standardization_prompt.to_json())

   # Load prompt template from JSON
    with open("../data_standardization_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt