import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # add project root to sys.path

from templates.prompt_template import BioinformaticsPrompt, FewShotExample

# Create a precision medicine prompt template
precision_medicine_prompt = BioinformaticsPrompt(
    research_area="Precision Medicine",
    description=(
        "Precision medicine utilizes molecular profiling, clinical data, and advanced analytics to tailor "
        "healthcare interventions to individual patients or patient subgroups. Bioinformatics plays a central "
        "role in integrating diverse data types—genomic, transcriptomic, proteomic, metabolomic, and clinical—to "
        "derive actionable insights for personalized diagnosis, prognosis, and treatment selection. This field "
        "has transformed cancer care through molecular subtyping and targeted therapy selection, and is expanding "
        "into other disease areas including rare genetic disorders, complex chronic conditions, and infectious "
        "diseases, ultimately aiming to optimize treatment outcomes while minimizing adverse effects through "
        "individualized approaches."
    ),
    key_concepts=[
        "Molecular profiling and biomarker identification",
        "Variant interpretation and clinical significance assessment",
        "Patient stratification and cohort discovery",
        "Pharmacogenomics and drug response prediction",
        "Treatment selection algorithms and clinical decision support",
        "Electronic health record integration and phenotyping",
        "Multi-modal data integration for personalized prediction",
        "Liquid biopsy and longitudinal monitoring"
    ],
    common_tools=[
        "ClinVar/CIViC - Clinical variant interpretation databases",
        "PharmGKB - Pharmacogenomics knowledge base",
        "cBioPortal - Cancer genomics data visualization",
        "OncoKB - Precision oncology knowledge base",
        "OMIM/Orphanet - Genetic disease databases",
        "GATK/VarScan - Variant calling and annotation",
        "PheWAS/R-PheWAS - Phenome-wide association studies",
        "g:Profiler/EnrichR - Pathway and functional enrichment"
    ],
    common_file_formats=[
        {"name": "VCF/gVCF", "description": "Variant call format for genomic variants"},
        {"name": "BAM/CRAM", "description": "Sequence alignment formats"},
        {"name": "MAF", "description": "Mutation Annotation Format for cancer variants"},
        {"name": "FHIR", "description": "Healthcare data exchange standard"},
        {"name": "DICOM", "description": "Medical imaging data format"},
        {"name": "PGx", "description": "Pharmacogenomic data exchange format"},
        {"name": "OMOP CDM", "description": "Common Data Model for EHR data"}
    ],
    examples=[
        FewShotExample(
            query="What are the key considerations for developing a molecular tumor board framework to guide precision oncology?",
            context="The user wants to establish a molecular tumor board at their institution to review complex cancer cases and make treatment recommendations based on molecular profiling.",
            response="""
# Molecular Tumor Board Framework for Precision Oncology

## Conceptual Framework

A molecular tumor board (MTB) synthesizes genomic, clinical, and pharmacological expertise to provide evidence-based treatment recommendations through:

1. **Systematic variant interpretation and prioritization**
2. **Clinical and molecular data integration**
3. **Evidence-based recommendation formulation**
4. **Treatment outcome tracking and learning**

An effective MTB balances scientific rigor with practical clinical application.

## Decision Framework

### Structural Organization Decisions
- **Board composition strategy**:
  - **Core expertise requirements**:
    - Medical oncologists (disease-specific vs. general)
    - Molecular pathologists
    - Clinical geneticists
    - Bioinformaticians/computational biologists
    - Pharmacists/pharmacologists
    - Research scientists
  
  - **Extended expertise considerations**:
    - Radiation oncologists
    - Surgeons
    - Genetic counselors
    - Patient advocates
    - Ethicists
    - Palliative care specialists
  
  - **Operational roles**:
    - Case coordinators
    - Variant curation specialists
    - Clinical trial navigators
    - Data managers

- **Meeting cadence and format**:
  - Frequency determination (weekly vs. biweekly)
  - Time allocation per case (standard vs. tiered approach)
  - Virtual vs. in-person vs. hybrid format
  - Synchronous vs. asynchronous review components
  - Emergency review protocols

- **Case selection criteria**:
  - Priority framework for case review
  - Refractory/advanced disease focus
  - Rare tumor types
  - Molecular complexity thresholds
  - Resource allocation for equity considerations

### Molecular Testing Framework
- **Testing strategy decisions**:
  - **Panel selection approach**:
    - Targeted panels vs. comprehensive genomic profiling
    - Single platform vs. multi-modal testing
    - Standard panels vs. disease-specific panels
    - Commercial vs. in-house assay development
  
  - **Specimen considerations**:
    - Tissue requirements and quality assessment
    - Fresh vs. archived sample utilization
    - Liquid biopsy integration strategy
    - Normal tissue requirements
    - Sequential biopsy protocols

- **Analytical pipeline design**:
  - Variant calling sensitivity/specificity balance
  - Tumor mutational burden calculation
  - Microsatellite instability assessment
  - Structural variant detection approach
  - Copy number alteration analysis
  - RNA fusion detection strategy

- **Turnaround time management**:
  - Expected timeframes for urgent vs. routine cases
  - Rate-limiting step identification
  - Parallel processing opportunities
  - Preliminary result reporting protocols
  - Communication of delays

### Variant Interpretation Approach
- **Variant prioritization framework**:
  - Driver vs. passenger discrimination
  - Actionability tiering system
  - Variant of unknown significance (VUS) management
  - Germline finding protocols
  - Incidental finding procedures
  
- **Evidence evaluation system**:
  - Level of evidence classification (e.g., OncoKB, CIViC tiers)
  - Literature curation methodology
  - Clinical trial matching criteria
  - Off-label use evaluation framework
  - Evidence strength assessment
  
- **Interpretation resources**:
  - Knowledge base selection and integration
  - In-house curation database development
  - Case repository maintenance
  - Interpretation consistency protocols
  - Reinterpretation triggers

### Treatment Recommendation Process
- **Recommendation formulation**:
  - Standard recommendation template design
  - Tiered therapeutic options presentation
  - Alternative treatment path inclusion
  - Clinical trial matching integration
  - Level of consensus documentation
  
- **Clinical integration mechanisms**:
  - EHR integration of recommendations
  - Clinical workflow embedding
  - Follow-up procedures
  - Treatment implementation barriers assessment
  - Patient-facing materials development
  
- **Decision support tools**:
  - Structured recommendation formats
  - Visual summary creation
  - Clinician-facing dashboards
  - Patient-oriented decision aids
  - Virtual molecular tumor board platforms

### Outcome Tracking and Learning
- **Follow-up tracking system**:
  - Treatment implementation rates
  - Response assessment standardization
  - Adverse event monitoring
  - Resistance mechanism identification
  - Time on therapy documentation
  
- **Database and registry design**:
  - Case repository structure
  - Outcome data elements
  - Data dictionary development
  - Longitudinal tracking mechanisms
  - Multi-institutional data sharing preparation
  
- **Quality improvement processes**:
  - Regular outcome review cadence
  - Recommendation efficacy assessment
  - Discrepancy analysis procedures
  - Process improvement mechanisms
  - Novel biomarker identification pathway

## Implementation Considerations

- **Start with focused scope**: Begin with specific tumor types or clinical scenarios
- **Implement phased rollout**: Pilot program before full implementation
- **Develop standardized workflows**: Create templates for case submission and review
- **Establish clear governance**: Define decision authority and escalation pathways
- **Plan for scalability**: Design systems that can grow with increasing case volumes

## Useful Resources

- [ASCO/CAP/AMP Guidelines](https://www.asco.org/practice-patients/guidelines) - Molecular biomarker testing guidelines
- [CIViC Knowledge Base](https://civicdb.org/) - Clinical interpretation of variants in cancer
- [OncoKB](https://www.oncokb.org/) - Precision oncology knowledge base
- [Cancer Molecular Tumor Board Series](https://ascopubs.org/doi/full/10.1200/PO.17.00011) - JCO case studies
- [Molecular Tumor Board Portal](https://www.mtbp.org/) - Virtual molecular tumor board platform
"""
        ),
        FewShotExample(
            query="How should I approach pharmacogenomic testing implementation in a clinical setting to guide medication decisions?",
            context="The user is a clinical pharmacist or physician looking to implement pharmacogenomic testing in their practice to guide medication selection and dosing.",
            response="""
# Pharmacogenomic Testing Implementation Framework

## Conceptual Framework

Clinical pharmacogenomics implementation requires a structured approach to:

1. **Patient and medication selection**: Identifying who to test and which medications to focus on
2. **Testing infrastructure development**: Establishing testing logistics and result delivery
3. **Clinical decision support integration**: Embedding PGx knowledge into workflow
4. **Educational program development**: Preparing clinicians and patients
5. **Outcome measurement**: Assessing impact and value

Success depends on balancing scientific evidence, clinical utility, and implementation feasibility.

## Decision Framework

### Program Scope Definition
- **Target population approach**:
  - **Universal vs. selective testing**:
    - Preemptive testing of all patients
    - Risk-based selection (polypharmacy, high-risk medications)
    - Disease-specific cohorts (psychiatry, cardiology, oncology)
    - Diagnostic vs. preventive orientation
  
  - **Demographic considerations**:
    - Ancestry-related variant frequency differences
    - Age-specific implementation strategies
    - Special populations (pediatrics, elderly, pregnant)
    - Health equity and access planning
  
  - **Clinical setting alignment**:
    - Primary care vs. specialty implementation
    - Inpatient vs. outpatient workflow differences
    - Urban vs. rural delivery adaptations
    - Point-of-care vs. scheduled testing

- **Gene-drug pair prioritization**:
  - CPIC/DPWG guideline availability
    - Level A vs. B recommendations
    - Actionability assessment
  - Medication utilization frequency
  - Adverse event severity and frequency
  - Alternative medication availability
  - Cost-effectiveness considerations

- **Panel composition decisions**:
  - Single gene assays vs. multi-gene panels
  - Core genes (CYP2C19, CYP2D6, CYP2C9, SLCO1B1, etc.)
  - Specialty-specific gene additions
  - SNP selection vs. full gene sequencing
  - Star allele vs. activity score reporting

### Testing Logistics Framework
- **Testing model selection**:
  - Send-out vs. in-house testing
  - CLIA/CAP certification requirements
  - Turnaround time requirements
  - Sample collection logistics
  - Cost and reimbursement considerations
  
- **Laboratory selection criteria**:
  - Analytical validity assessment
  - Test comprehensiveness and limitations
  - Result format and interpretability
  - EHR integration capabilities
  - Cost and insurance coverage

- **Sample collection protocol**:
  - Blood vs. saliva collection
  - Sample handling procedures
  - Chain of custody documentation
  - Interfering factors management
  - Rejection criteria and handling

- **Result reporting standardization**:
  - Genotype vs. phenotype representation
  - Star allele nomenclature usage
  - Medication-specific interpretations
  - Report design for clinician usability
  - Timeline for result availability

### Clinical Decision Support Integration
- **Alert and notification design**:
  - Active vs. passive alert strategies
  - Alert timing (at prescribing vs. pre-encounter)
  - Alert design to minimize fatigue
  - Override documentation requirements
  - Critical vs. informational distinction
  
- **EHR integration approach**:
  - Results storage location (lab vs. problem list vs. dedicated section)
  - SMART on FHIR app vs. native EHR functionality
  - Interoperability with external PGx resources
  - Order set and prescription modification
  - Documentation template development

- **Recommendation format decisions**:
  - Gene-based vs. medication-based organization
  - Alternative medication suggestions
  - Dosing adjustment specificity
  - Supporting evidence presentation
  - Consultation recommendation criteria

- **Clinical workflow embedding**:
  - Pre-visit planning processes
  - Medication reconciliation integration
  - Pharmacy review triggers
  - Consultation service development
  - Refill management protocol updates

### Education and Support Program
- **Provider education strategy**:
  - Targeted vs. general educational approach
  - Knowledge assessment and gaps identification
  - Just-in-time vs. comprehensive education
  - Specialty-specific training modules
  - Ongoing education and updates mechanism
  
- **Patient education materials**:
  - Health literacy-appropriate explanations
  - Cultural competency considerations
  - Visual aids and decision support tools
  - Result interpretation guidance
  - Family implications discussion

- **Genetic counseling integration**:
  - Referral criteria for genetic counseling
  - Pre-test vs. post-test counseling roles
  - Direct vs. indirect involvement models
  - Remote counseling options
  - Incidental finding management

- **Support infrastructure**:
  - Implementation team composition
  - Clinical champion identification
  - Consultation service development
  - Question/issue escalation pathway
  - Ongoing program maintenance responsibilities

### Outcome Measurement and Evaluation
- **Metrics selection**:
  - Process measures (testing rates, alert override rates)
  - Outcome measures (adverse events, hospitalizations)
  - Economic measures (cost avoidance, resource utilization)
  - Patient-reported outcomes
  - Implementation success metrics
  
- **Data collection strategies**:
  - Prospective vs. retrospective assessment
  - Registry development
  - Automated vs. manual data collection
  - Comparison group identification
  - Longitudinal tracking approach

- **Quality improvement process**:
  - PDSA cycle implementation
  - Alert optimization procedure
  - Regular guideline update integration
  - User feedback incorporation
  - Continuous improvement methodology

## Implementation Considerations

- **Start small and focused**: Begin with high-impact gene-drug pairs
- **Engage stakeholders early**: Include pharmacy, IT, clinicians, and administration
- **Create streamlined workflows**: Minimize disruption to clinical practice
- **Prepare for incidental findings**: Develop protocols for unexpected results
- **Plan for sustainability**: Address long-term funding and maintenance

## Useful Resources

- [CPIC Guidelines](https://cpicpgx.org/) - Clinical Pharmacogenetics Implementation Consortium
- [PharmGKB](https://www.pharmgkb.org/) - Pharmacogenomics knowledge base
- [CDC's Genomics Implementation Toolkit](https://www.cdc.gov/genomics/implementation/toolkit/index.htm)
- [NIH Pharmacogenomics Research Network](https://www.pgrn.org/)
- [IGNITE Network Implementation Resources](https://ignite-genomics.org/)
"""
        )
    ],
    references=[
        "Manolio TA, et al. (2022). The Clinical Imperative for Includability, Equity, and Diversity in Clinical Genomics Research. The American Journal of Human Genetics, 109(6), 959-965.",
        "Seyhan AA, Carini C. (2023). Biomarkers in drug discovery and development. Nature Reviews Drug Discovery, 22(1), 43-61.",
        "McDonough CW, et al. (2023). Pharmacogenomics at the point of care: a community pharmacy implementation. NPJ Genomic Medicine, 8(1), 15.",
        "Dias R, Torkamani A. (2022). Artificial intelligence in clinical and genomic diagnostics. Genome Medicine, 14(1), 14.",
        "Bai R, et al. (2024). Using genetics to guide clinical decisions: five considerations from the eMERGE Network. BMC Medicine, 22(1), 61."
    ]
)

# Save prompt template to JSON
if __name__ == "__main__":
    # Test with a sample query
    user_query = "What are the best practices for interpreting and reporting incidental findings from whole genome sequencing?"
    
    # Generate prompt
    prompt = precision_medicine_prompt.generate_prompt(user_query)
    print(prompt)
    
    # Save prompt template to JSON
    with open("../precision_medicine_prompt.json", "w") as f:
        f.write(precision_medicine_prompt.to_json())

    # Load prompt template from JSON
    with open("../precision_medicine_prompt.json", "r") as f:
        loaded_prompt = BioinformaticsPrompt.from_json(f.read())
    
    # Verify it works the same
    assert loaded_prompt.generate_prompt(user_query) == prompt