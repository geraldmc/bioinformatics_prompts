import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a blockchain in bioinformatics prompt template
blockchain_bioinformatics_prompt = BioinformaticsPrompt(
    discipline="Blockchain Technology in Bioinformatics",
    description=(
        "Blockchain technology offers novel approaches to address critical challenges in bioinformatics data "
        "management, including security, privacy, provenance tracking, and controlled sharing. By providing "
        "immutable, transparent, and decentralized ledgers, blockchain can enhance data integrity, enable "
        "fine-grained access control, and create auditable trails of data usage. These capabilities are "
        "particularly valuable for sensitive genomic data, collaborative research platforms, and clinical "
        "data integration where trust, privacy, and regulatory compliance are paramount concerns. As "
        "bioinformatics increasingly deals with large-scale, multi-institutional datasets, blockchain "
        "offers frameworks for balancing openness with security and individual privacy with research utility."
    ),
    key_concepts=[
        "Distributed ledger technology for biological data",
        "Smart contracts for automated data access and usage policies",
        "Tokenization of biological assets and data access rights",
        "Genomic data privacy and consent management",
        "Immutable audit trails for bioinformatics workflows",
        "Decentralized biomedical research collaboration",
        "Self-sovereign identity for research participants",
        "Blockchain consensus mechanisms for biomedical applications"
    ],
    common_tools=[
        "Hyperledger Fabric - Permissioned blockchain framework",
        "IPFS - Distributed file system for large dataset storage",
        "Ethereum - Smart contract platform",
        "OpenZeppelin - Secure smart contract development",
        "BigchainDB - Scalable blockchain database",
        "Solid (Social Linked Data) - Decentralized data storage",
        "Oasis Network - Privacy-preserving computation",
        "Ocean Protocol - Decentralized data exchange protocol"
    ],
    common_file_formats=[
        {"name": "JSON-LD", "description": "Linked data format for blockchain metadata"},
        {"name": "JWS/JWT", "description": "JSON Web Signatures/Tokens for verifiable credentials"},
        {"name": "Solidity", "description": "Smart contract programming language files"},
        {"name": "CID", "description": "Content identifiers for IPFS-stored data"},
        {"name": "DID", "description": "Decentralized identifier documents"},
        {"name": "FHIR", "description": "Healthcare interoperability format for blockchain applications"},
        {"name": "BAM/VCF with hash", "description": "Standard genomic formats with integrity hashes"}
    ],
    examples=[
        FewShotExample(
            query="How should I design a blockchain-based system for secure sharing of genomic data across multiple research institutions?",
            context="The user wants to implement a blockchain solution to enable controlled sharing of genomic datasets between multiple research institutions while ensuring privacy, consent enforcement, and data integrity.",
            response="""
# Blockchain-Based Genomic Data Sharing Architecture

## Conceptual Framework

A blockchain-based genomic data sharing system must address these core challenges:

1. **Privacy preservation** while enabling research utility
2. **Scalable storage** of large genomic datasets
3. **Fine-grained access control** with verifiable consent
4. **Regulatory compliance** across jurisdictions
5. **Computational efficiency** for genomic analysis

The architecture requires balancing on-chain (metadata, access controls) and off-chain (raw data) components.

## Decision Framework

### Blockchain Architecture Selection
- **Platform type decision**:
  - **Public blockchain** (Ethereum, Cardano):
    - Advantages: Maximum transparency, censorship resistance
    - Limitations: Privacy concerns, scalability issues, higher costs
  
  - **Permissioned blockchain** (Hyperledger Fabric, Quorum):
    - Advantages: Higher throughput, known validators, access controls
    - Limitations: More centralized, requires governance structure
  
  - **Hybrid approaches**:
    - Public chain for verification, private for transactions
    - Sidechains for specific functionality
    - Layer-2 solutions for scalability

- **Consensus mechanism selection**:
  - Proof of Work: High security but environmental concerns
  - Proof of Stake: More efficient but potential centralization
  - Proof of Authority: Efficient for permissioned networks
  - PBFT variants: Deterministic finality for healthcare applications
  - Selection impact on throughput, finality, and energy consumption

- **On-chain vs. off-chain storage strategy**:
  - What belongs on-chain: access logs, consent receipts, data pointers
  - What belongs off-chain: raw genomic data, large result sets
  - Hybrid storage options: header on-chain, data in IPFS/Filecoin

### Data Privacy Implementation
- **Privacy preservation approach**:
  - **Zero-knowledge proofs**: For verifiable computation without data exposure
  - **Homomorphic encryption**: For computation on encrypted data
  - **Secure multi-party computation**: For distributed analysis
  - **Differential privacy**: For aggregate analysis with formal privacy guarantees
  - Appropriateness depends on analysis types and sensitivity requirements

- **Data granularity and access control**:
  - Whole genome vs. specific regions vs. variant-level access
  - Access tiers (open, controlled, restricted)
  - Temporary vs. permanent access grants
  - Emergency access provisions
  - Re-identification risk assessment framework

- **Identity and authentication system**:
  - Self-sovereign identity vs. federated identity
  - Researcher credentials and institutional verification
  - Separation between identity and pseudonymous data access
  - Multi-signature requirements for sensitive data

### Consent Management Design
- **Consent representation**:
  - Machine-readable consent format selection
  - Ontology-based consent encoding
  - Dynamic vs. static consent models
  - Consent chain-of-custody tracking

- **Smart contract consent enforcement**:
  - Automated policy execution
  - Time-bound access implementation
  - Usage limitation enforcement (research purpose restrictions)
  - Withdrawal of consent handling
  - Regulatory updates accommodation

- **Patient/donor engagement model**:
  - Consent visualization and verification
  - Usage notifications and alerts
  - Benefit sharing mechanisms
  - Re-contact mechanisms for additional consent

### Data Integrity and Provenance
- **Data verification mechanism**:
  - Whole file hashing vs. segmented approaches
  - Hash storage strategy (Merkle trees, direct hashes)
  - Tamper detection mechanisms
  - Frequency of integrity verification

- **Provenance tracking granularity**:
  - Dataset-level vs. record-level tracking
  - Workflow step documentation
  - Analysis parameter preservation
  - Result verification approach

- **Audit capabilities**:
  - Real-time vs. retrospective auditing
  - Automated compliance checking
  - Anomaly detection in access patterns
  - Regulatory reporting automation

### Implementation Considerations
- **Interoperability framework**:
  - Standards adoption (GA4GH, FHIR, HL7)
  - API design for external system integration
  - Legacy system compatibility
  - Cross-chain interoperability

- **Governance structure**:
  - Centralized vs. decentralized governance
  - Stakeholder representation
  - Protocol update mechanisms
  - Dispute resolution processes
  - Incentive alignment for participants

- **Scalability and performance strategy**:
  - Throughput requirements assessment
  - Sharding or partitioning approach
  - Caching strategy for frequent queries
  - Transaction batching optimization

## Key Implementation Considerations

- **Start with clear data governance policies**: Blockchain enforces but doesn't create policy
- **Validate with small proof-of-concept**: Test with non-sensitive data first
- **Consider regulatory landscape**: Ensure compliance with GDPR, HIPAA, etc.
- **Plan for technical sustainability**: Consider long-term maintenance and updates
- **Implement progressive security measures**: Layer security from data to network levels

## Useful Resources

- [GA4GH Data Use Ontology](https://github.com/EBISPOT/DUO) - Standardized terms for data use conditions
- [Hyperledger Healthcare SIG](https://wiki.hyperledger.org/display/HCSIG/Healthcare+SIG) - Industry-specific implementations
- [BMC Medical Genomics: Blockchain in Genomics](https://bmcmedgenomics.biomedcentral.com/articles/collections/blockchaingenomics)
- [HL7 FHIR Genomics Implementation Guide](https://hl7.org/fhir/genomics.html)
- [European Blockchain Partnership](https://digital-strategy.ec.europa.eu/en/policies/blockchain-partnership)
"""
        ),
        FewShotExample(
            query="What are the key considerations for implementing blockchain-based provenance tracking for bioinformatics workflows?",
            context="The user wants to implement a system that uses blockchain to track the complete history of bioinformatics analyses, ensuring reproducibility and transparent documentation of all processing steps.",
            response="""
# Blockchain-Based Bioinformatics Workflow Provenance

## Conceptual Framework

Provenance tracking in bioinformatics requires documenting:

1. **Input data origins and versions**
2. **Processing steps and parameters**
3. **Computational environment details**
4. **Output data transformations**
5. **User actions and decisions**

Blockchain can create an immutable, time-stamped record of these elements while maintaining data security and accessibility.

## Decision Framework

### Provenance Granularity Selection
- **Entity-level granularity decisions**:
  - **Fine-grained tracking** (individual files, operations):
    - Advantages: Complete audit trail, maximum reproducibility
    - Limitations: Performance overhead, storage requirements
  
  - **Coarse-grained tracking** (workflow runs, major steps):
    - Advantages: Lower overhead, simpler implementation
    - Limitations: Less detailed provenance, some opacity
  
  - **Multi-level approach**:
    - Critical operations tracked in detail
    - Routine operations tracked at higher level
    - Based on sensitivity, regulatory requirements, and reproducibility needs

- **Versioning strategy selection**:
  - Content-addressed storage vs. incremental versioning
  - Full history vs. state transitions
  - Snapshot frequency determination
  - Branch and merge handling for parallel analyses

- **Provenance query requirements**:
  - Temporal queries (as-of-time views)
  - Lineage queries (upstream/downstream tracing)
  - Attribution queries (who did what when)
  - Filtering and aggregation capabilities

### Blockchain Implementation Strategy
- **On-chain vs. off-chain storage balance**:
  - **On-chain elements**:
    - Cryptographic hashes of datasets
    - Workflow execution metadata
    - Access and modification records
    - Data usage commitments
  
  - **Off-chain storage options**:
    - IPFS/Filecoin for distributed storage
    - Secure institutional repositories
    - Specialized scientific data stores
    - Encrypted cloud storage with blockchain verification

- **Smart contract functionality**:
  - Provenance registration and validation
  - Automated verification of workflow steps
  - Access control enforcement
  - Conditional data release based on requirements
  - Integration with compute environments

- **Consensus mechanism appropriateness**:
  - Write frequency requirements
  - Finality needs for regulatory compliance
  - Energy efficiency considerations
  - Validation node distribution and trust model

### Scientific Workflow Integration
- **Workflow system compatibility**:
  - Integration with existing tools (Nextflow, Snakemake, Galaxy)
  - Modification vs. wrapper approaches
  - Transparent vs. user-initiated recording
  - Backward compatibility with existing workflows

- **Runtime environment documentation**:
  - Container hash verification (Docker, Singularity)
  - Library and dependency versioning
  - Hardware configuration recording
  - Parameter space documentation
  - Random seed and stochastic process handling

- **Data transformation documentation**:
  - Intermediate file tracking strategy
  - In-memory transformation documentation
  - Quality control metrics recording
  - Filtering decisions and thresholds
  - Statistical methods and parameters

### Validation and Compliance Features
- **Reproducibility verification approach**:
  - Re-execution capabilities from blockchain records
  - Partial vs. complete workflow reproduction
  - Equivalence checking for results
  - Deviation detection and alerting
  - Time-travel debugging capabilities

- **Regulatory compliance features**:
  - 21 CFR Part 11 requirements for electronic records
  - GDPR data processing documentation
  - FAIR principles implementation
  - Domain-specific compliance (HIPAA, GxP)
  - Audit preparation automation

- **Trust and validation mechanisms**:
  - Multi-party validation of critical results
  - Institutional endorsements and signatures
  - Pre-registration of analysis plans
  - Blind analysis support
  - Independent verification protocols

### Practical Implementation Decisions
- **User experience design**:
  - Transparency vs. complexity balance
  - Visualization of provenance graphs
  - Notification and alerting systems
  - Permission management interfaces
  - Adoption barrier minimization

- **Performance optimization strategy**:
  - Batching of provenance transactions
  - Selective recording based on criticality
  - Asynchronous verification options
  - Caching and indexing strategies
  - Scalability planning for large workflows

- **Governance and sustainability model**:
  - Stewardship of the provenance chain
  - Node operation responsibilities
  - Funding model for infrastructure
  - Protocol update mechanisms
  - Long-term accessibility planning (10+ years)

## Implementation Considerations

- **Start with critical workflows**: Focus on high-value analyses requiring rigorous documentation
- **Layer into existing systems**: Integrate with current workflow tools rather than replacing them
- **Consider domain-specific requirements**: Different fields have varying provenance needs
- **Plan for interoperability**: Ensure provenance records can be exchanged between systems
- **Address cultural adoption**: Provide incentives and minimize friction for researcher participation

## Useful Resources

- [W3C PROV Data Model](https://www.w3.org/TR/prov-dm/) - Standard for provenance interchange
- [BioCompute Objects](https://biocomputeobject.org/) - FDA-recognized standard for workflow provenance
- [Research Object Crate](https://www.researchobject.org/ro-crate/) - Packaging research artifacts with metadata
- [Blockchain for Science](https://www.blockchainforscience.com/) - Community focused on blockchain in research
- [DataLad](https://www.datalad.org/) - Distributed data management that could integrate with blockchain
"""
        )
    ],
    references=[
        "Ozercan HI, et al. (2021). Realizing the potential of blockchain technologies in genomics. Genome Biology, 22(1), 1-10.",
        "Gursoy G, et al. (2023). Privacy-preserving data sharing using blockchain and federated learning in genomics. Nature Computational Science, 3(6), 491-504.",
        "Kuo TT, et al. (2022). Blockchain distributed ledger technologies for biomedical and health care applications. Journal of the American Medical Informatics Association, 29(1), 158-169.",
        "Leeming G, et al. (2023). Securing research data: a review of blockchain in biomedical informatics. BMC Medical Informatics and Decision Making, 23(1), 88.",
        "Agbo CC, et al. (2022). A comprehensive review of blockchain technology in clinical trials management for biomedical research. IEEE Access, 10, 105825-105842."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "How can I implement blockchain to ensure privacy and security of patient genomic data?"
    
    # Generate prompt
    prompt = blockchain_bioinformatics_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../blockchain_bioinformatics_prompt.json", "w") as f:
        f.write(blockchain_bioinformatics_prompt.to_json())

    # Load prompt template from JSON
    with open("../blockchain_bioinformatics_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt