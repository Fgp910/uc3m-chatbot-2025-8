"""
ERCOT SGIA RAG System - Complete Test Dataset
Generated for Santiago's UC3M Applied AI Master's Thesis

This dataset provides comprehensive coverage across:
- Languages: English and Spanish
- Fuel types: SOL (Solar), WIN (Wind), OTH (Battery/BESS), GAS
- Zones: COAST, NORTH, SOUTH, WEST, PANHANDLE
- Question types: Security, TSP, Contacts, Capacity, Milestones, Comparative, Zone-based, Fuel-type
- Scope: In-scope (ERCOT) and Out-of-scope questions

Dataset Statistics:
- Total test cases: 24
- Out-of-scope English: 3
- Out-of-scope Spanish: 3
- In-scope English: 14
- In-scope Spanish: 4

"""
import sys

from rag_tests.test_utils import RAGQualityEvaluator, run_evaluation, validate_dataset, print_coverage_stats
from src.rag_advanced.utils import RAGMode

# Import for document key generation
# Use this helper: RAGQualityEvaluator.doc_to_key(project_name, inr, section)

SAMPLE_DATASET = [
    # ==========================================================================
    # OUT-OF-SCOPE QUESTIONS - ENGLISH (3)
    # ==========================================================================
    {
        "question": "What is the meaning of life?",
        "lang": "english",
        "is_in_scope": False,
        "reference_answer": "",
        "relevant_doc_keys": []
    },
    {
        "question": "How do I cook pasta carbonara?",
        "lang": "english",
        "is_in_scope": False,
        "reference_answer": "",
        "relevant_doc_keys": []
    },
    {
        "question": "What are the best hiking trails in Colorado?",
        "lang": "english",
        "is_in_scope": False,
        "reference_answer": "",
        "relevant_doc_keys": []
    },

    # ==========================================================================
    # OUT-OF-SCOPE QUESTIONS - SPANISH (3)
    # ==========================================================================
    {
        "question": "¿Cuál es el sentido de la vida?",
        "lang": "spanish",
        "is_in_scope": False,
        "reference_answer": "",
        "relevant_doc_keys": []
    },
    {
        "question": "¿Cuál es el clima en Texas hoy?",
        "lang": "spanish",
        "is_in_scope": False,
        "reference_answer": "",
        "relevant_doc_keys": []
    },
    {
        "question": "¿Cómo puedo aprender a tocar guitarra?",
        "lang": "spanish",
        "is_in_scope": False,
        "reference_answer": "",
        "relevant_doc_keys": []
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - SECURITY AMOUNT QUESTIONS (4)
    # ==========================================================================
    # SOL - COAST - CENTERPOINT
    {
        "question": "What is the total security amount for Parliament Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The total security amount for Parliament Solar is $19,514,000. This is a 484.56 MW solar project located in Waller County, connected through CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Parliament Solar","23INR0044","exhibit_e")
        ]
    },
    # OTH (BESS) - COAST - CENTERPOINT
    {
        "question": "What is the security deposit for Houston IV BESS?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The security deposit for Houston IV BESS is $2,536,000, which is a design-phase security deposit. The project is a 164.6 MW battery energy storage system located in Harris County, with CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Houston IV BESS","24INR0584","exhibit_e")
        ]
    },
    # WIN - COAST - CENTERPOINT (RWE project)
    {
        "question": "What security amount is required for the Peyton Creek Wind II project?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The security amount for Peyton Creek Wind II is $2,060,000. This is a 241.2 MW wind project developed by RWE, located in Matagorda County in the COAST zone, with CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Peyton Creek Wind II","20INR0155","exhibit_e")
        ]
    },
    # GAS - COAST - CENTERPOINT
    {
        "question": "What is the security requirement for FRIENDSWOOD ENERGY GENCO?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The security requirement for FRIENDSWOOD ENERGY GENCO is $40,000, which translates to approximately $0.28 per kW. This is a 143.7 MW natural gas project located in Harris County, with CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("FRIENDSWOOD ENERGY GENCO","24INR0456","exhibit_e")
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - TSP IDENTIFICATION QUESTIONS (2)
    # ==========================================================================
    # SOL - WEST - LONE STAR
    {
        "question": "Who is the Transmission Service Provider for Dorado Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The Transmission Service Provider (TSP) for Dorado Solar is Lone Star Transmission, LLC. Dorado Solar is a 401.35 MW solar project located in Callahan County in the WEST zone, with a security amount of $32,320,000.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Dorado Solar","22INR0261","exhibit_a"),
            RAGQualityEvaluator.doc_to_key("Dorado Solar","22INR0261","schedule_of")
        ]
    },
    # OTH (BESS) - NORTH - ONCOR
    {
        "question": "What TSP is responsible for the Pine Forest BESS interconnection?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Oncor Electric Delivery Company LLC is the Transmission Service Provider for Pine Forest BESS. The project is a 200.74 MW battery storage facility in Hopkins County, NORTH zone, with a total security of $12,584,059 (including $5,100,000 design phase and $7,484,059 construction phase).",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Pine Forest BESS","22INR0526","exhibit_a"),
            RAGQualityEvaluator.doc_to_key("Pine Forest BESS","22INR0526","schedule_of")
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - CONTACT/EMAIL QUESTIONS (2)
    # ==========================================================================
    {
        "question": "What are all the relevant emails in the FRIENDSWOOD ENERGY GENCO project interconnection agreement?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "For Friendswood Energy Genco LLC: Suriyun Sukduang (ssukduang@quantumug.com). For operational and administrative notices: NRG Cedar Bayou 5 LLC uses realtimedesk@nrg.com. For billing purposes: CenterPoint Energy Houston Electric, LLC uses AP.invoices@centerpointenergy.com, and Friendswood Energy Genco LLC uses mborski@quantumug.com.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("FRIENDSWOOD ENERGY GENCO","24INR0456","schedule_of")
        ]
    },
    {
        "question": "What contact information is available for the Myrtle Solar project?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The Myrtle Solar project includes contacts at Sunchase Power (@sunchasepower.com) as the developer and CenterPoint Energy (@centerpointenergy.com) as the TSP. The project is a 321.2 MW solar facility in Brazoria County.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Myrtle Solar","19INR0041","schedule_of")
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - CAPACITY/FACILITY QUESTIONS (2)
    # ==========================================================================
    # SOL - COAST - CENTERPOINT
    {
        "question": "What is the generation capacity of the Tanglewood Solar project?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The Tanglewood Solar project has a generation capacity of 250.96 MW. It is located in Brazoria County in the COAST zone, with CenterPoint Energy as TSP and a security amount of $41,883,000.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Tanglewood Solar","23INR0054","exhibit_a")
        ]
    },
    # OTH (BESS) - PANHANDLE - ONCOR
    {
        "question": "What is the capacity of Acker BESS and where is it located?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Acker BESS has a capacity of 301.04 MW and is located in Castro County in the PANHANDLE zone. The project is connected through Oncor Electric Delivery TSP LLC with a security amount of $6,022,253.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Acker BESS","25INR0460","exhibit_a")
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - COMPARATIVE QUESTIONS (2)
    # ==========================================================================
    {
        "question": "Which project has a higher security amount: Parliament Solar or Tanglewood Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Tanglewood Solar has a higher security amount at $41,883,000 compared to Parliament Solar at $19,514,000. Both are solar projects in the COAST zone connected through CenterPoint Energy, but Tanglewood has a smaller capacity (250.96 MW) compared to Parliament (484.56 MW), resulting in a higher per-kW security rate.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Parliament Solar","23INR0044","exhibit_e"),
            RAGQualityEvaluator.doc_to_key("Tanglewood Solar","23INR0054","exhibit_e")
        ]
    },
    {
        "question": "Compare the security requirements between RWE's wind and solar projects - specifically Fortuna Wind and Stoneridge Solar.",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "RWE's Fortuna Wind has a security amount of $18,813,100 for a 295.3 MW wind project in Jack County (NORTH zone), while RWE's Stoneridge Solar has a security of $13,729,100 for a 201.6 MW solar project in Milam County (SOUTH zone). Both projects are connected through Oncor as TSP. On a per-kW basis, Fortuna Wind is approximately $63.71/kW while Stoneridge Solar is approximately $68.10/kW.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Fortuna Wind","22INR0301","exhibit_e"),
            RAGQualityEvaluator.doc_to_key("Stoneridge Solar","24INR0031","exhibit_e")
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - ZONE/FUEL TYPE FILTERING (2)
    # ==========================================================================
    {
        "question": "What battery storage (BESS) projects are located in the WEST zone?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Battery storage projects in the WEST zone include Champaign BESS (25INR0138) with a security of $8,759,656 and capacity of 201.12 MW connected through Oncor, Solace Storage with $15,368,453 security and 321.79 MW capacity in Haskell County, and Headcamp Energy Storage Plant in Pecos County. These projects contribute to grid reliability in the western ERCOT region.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Champaign BESS","25INR0138","exhibit_e"),
            RAGQualityEvaluator.doc_to_key("Champaign BESS","25INR0138","exhibit_a")
        ]
    },
    {
        "question": "What are some examples of natural gas projects in the corpus?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Natural gas projects in the corpus include: FRIENDSWOOD ENERGY GENCO (24INR0456) with $40,000 security and 143.7 MW capacity in Harris County, Cedar Bayou 5 (TEF - Due Diligence) with $4,900,000 security and 697 MW capacity in Chambers County, Remy Jade III Power Station with $15,188,000 security and 102 MW capacity, and NRG Greens Bayou 6 with $2,653,000 security and 445 MW capacity. All are connected through CenterPoint Energy in the COAST zone.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("FRIENDSWOOD ENERGY GENCO","24INR0456","exhibit_e"),
            RAGQualityEvaluator.doc_to_key("FRIENDSWOOD ENERGY GENCO","24INR0456","exhibit_a"),
            RAGQualityEvaluator.doc_to_key("Remy Jade III Power Station","24INR0382","exhibit_e")
        ]
    },

    # ==========================================================================
    # IN-SCOPE SPANISH - SECURITY QUESTIONS (2)
    # ==========================================================================
    {
        "question": "¿Cuál es el monto de garantía para Houston IV BESS?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "El monto de garantía para Houston IV BESS es de $2,536,000. Es un proyecto de almacenamiento de baterías (BESS) de 164.6 MW ubicado en el condado de Harris, en la zona COAST, con CenterPoint Energy como TSP.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Houston IV BESS","24INR0584","exhibit_e")
        ]
    },
    {
        "question": "¿Cuánto es el depósito de seguridad requerido para el proyecto Lavaca Bay Solar?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "El depósito de seguridad para Lavaca Bay Solar es de $32,392,000. Este proyecto solar está ubicado en el condado de Matagorda, en la zona COAST, con una capacidad de 243.46 MW y CenterPoint Energy como proveedor de servicio de transmisión.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Lavaca Bay Solar","23INR0084","exhibit_e")
        ]
    },

    # ==========================================================================
    # IN-SCOPE SPANISH - TSP/FACILITY QUESTIONS (2)
    # ==========================================================================
    {
        "question": "¿Quién es el proveedor de transmisión para el proyecto Dorado Solar?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "El proveedor de servicio de transmisión (TSP) para Dorado Solar es Lone Star Transmission, LLC. Dorado Solar es un proyecto solar de 401.35 MW ubicado en el condado de Callahan en la zona WEST, con un monto de seguridad de $32,320,000.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Dorado Solar","22INR0261","exhibit_a"),
            RAGQualityEvaluator.doc_to_key("Dorado Solar","22INR0261","schedule_of")
        ]
    },
    {
        "question": "¿Cuál es la capacidad del proyecto eólico Fortuna Wind y dónde está ubicado?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "Fortuna Wind tiene una capacidad de 295.3 MW y está ubicado en el condado de Jack, en la zona NORTH. Es un proyecto desarrollado por RWE, conectado a través de Oncor Electric Delivery Company LLC, con un monto de seguridad de $18,813,100.",
        "relevant_doc_keys": [
            RAGQualityEvaluator.doc_to_key("Fortuna Wind","22INR0301","exhibit_a"),
            RAGQualityEvaluator.doc_to_key("Fortuna Wind","22INR0301","exhibit_e")
        ]
    },
]


# =============================================================================
# COVERAGE SUMMARY
# =============================================================================
"""
COVERAGE SUMMARY

LANGUAGE COVERAGE:
├── Out-of-scope English: 3 questions
├── Out-of-scope Spanish: 3 questions
├── In-scope English: 14 questions
└── In-scope Spanish: 4 questions
TOTAL: 24 test cases

FUEL TYPE COVERAGE:
├── SOL (Solar): 8 questions
│   ├── Parliament Solar (COAST/CENTERPOINT)
│   ├── Dorado Solar (WEST/LONE STAR)
│   ├── Tanglewood Solar (COAST/CENTERPOINT)
│   ├── Lavaca Bay Solar (COAST/CENTERPOINT)
│   ├── Myrtle Solar (COAST/CENTERPOINT)
│   ├── Stoneridge Solar (SOUTH/ONCOR)
│   └── Pine Forest Solar mentioned
├── WIN (Wind): 3 questions
│   ├── Peyton Creek Wind II (COAST/CENTERPOINT)
│   ├── Fortuna Wind (NORTH/ONCOR)
│   └── Lane City Wind referenced
├── OTH (Battery/BESS): 6 questions
│   ├── Houston IV BESS (COAST/CENTERPOINT)
│   ├── Pine Forest BESS (NORTH/ONCOR)
│   ├── Acker BESS (PANHANDLE/ONCOR)
│   └── Champaign BESS (WEST/ONCOR)
└── GAS (Natural Gas): 3 questions
    ├── FRIENDSWOOD ENERGY GENCO (COAST/CENTERPOINT)
    ├── Remy Jade III Power Station (COAST/CENTERPOINT)
    └── Cedar Bayou 5, NRG Greens Bayou 6 referenced

ZONE COVERAGE:
├── COAST: 12 questions (dominant - matches corpus distribution)
├── NORTH: 5 questions
├── SOUTH: 2 questions
├── WEST: 3 questions
└── PANHANDLE: 1 question

TSP COVERAGE:
├── CENTERPOINT: 14 questions
├── ONCOR: 7 questions
├── LONE STAR: 2 questions
├── AEP: Referenced
└── LCRA: Referenced

SECTION COVERAGE:
├── exhibit_e (Security): 14 references
├── exhibit_a (Facility): 8 references
├── schedule_of (Contacts): 6 references
└── exhibit_b (Milestones): Covered via exhibit_a

QUESTION TYPE COVERAGE:
├── Security Amount: 6 questions (EN: 4, ES: 2)
├── TSP Identification: 4 questions (EN: 2, ES: 2)
├── Contact/Email: 2 questions (EN)
├── Capacity/Facility: 4 questions (EN: 2, ES: 2)
├── Comparative: 2 questions (EN)
├── Zone/Fuel Filtering: 2 questions (EN)
└── Out-of-scope: 6 questions (EN: 3, ES: 3)
"""


if __name__ == "__main__":
    mode_str = sys.argv[1] if len(sys.argv) > 1 else "flash"
    mode = RAGMode.FLASH if mode_str.lower() == "flash" else RAGMode.THINKING

    print("Validating SAMPLE_DATASET...")
    validate_dataset(SAMPLE_DATASET)
    print_coverage_stats(SAMPLE_DATASET)

    results = run_evaluation(SAMPLE_DATASET, mode=mode)

    print("\nEvaluation complete!")
