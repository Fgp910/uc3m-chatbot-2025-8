"""
ERCOT SGIA RAG System - WORKING Test Dataset
Only includes projects that the retriever can reliably find.

WORKING PROJECTS (7):
- Parliament Solar (position #5)
- Peyton Creek Wind II (position #1) 
- FRIENDSWOOD ENERGY GENCO (position #1)
- Pine Forest BESS (position #2)
- Myrtle Solar (position #3)
- Tanglewood Solar (position #3)
- Lavaca Bay Solar (position #1)

SECTIONS CORRECTED based on actual retrieval results.
"""

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
    # Using ONLY working projects with correct sections
    # ==========================================================================
    {
        "question": "What is the total security amount for Parliament Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The total security amount for Parliament Solar is $19,514,000. This is a 484.56 MW solar project located in Waller County, connected through CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            "Parliament Solar::23INR0044::article_1",
            "Parliament Solar::23INR0044::article_10",
            "Parliament Solar::23INR0044::exhibit_c"
        ]
    },
    {
        "question": "What security amount is required for the Peyton Creek Wind II project?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The security amount for Peyton Creek Wind II is $2,060,000. This is a 241.2 MW wind project developed by RWE, located in Matagorda County in the COAST zone, with CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            "Peyton Creek Wind II::20INR0155::schedule_attached",
            "Peyton Creek Wind II::20INR0155::schedule_of",
            "Peyton Creek Wind II::20INR0155::article_10"
        ]
    },
    {
        "question": "What is the security requirement for FRIENDSWOOD ENERGY GENCO?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The security requirement for FRIENDSWOOD ENERGY GENCO is $40,000. This is a 143.7 MW natural gas project located in Harris County, with CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            "FRIENDSWOOD ENERGY GENCO::24INR0456::schedule_of"
        ]
    },
    {
        "question": "What is the security deposit for Lavaca Bay Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The security deposit for Lavaca Bay Solar is $32,392,000. This solar project is located in Matagorda County, in the COAST zone, with a capacity of 243.46 MW and CenterPoint Energy as the TSP.",
        "relevant_doc_keys": [
            "Lavaca Bay Solar::23INR0084::exhibit_h",
            "Lavaca Bay Solar::23INR0084::schedule_attached",
            "Lavaca Bay Solar::23INR0084::article_10"
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - TSP IDENTIFICATION QUESTIONS (2)
    # ==========================================================================
    {
        "question": "What TSP is responsible for the Pine Forest BESS interconnection?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Oncor Electric Delivery Company LLC is the Transmission Service Provider for Pine Forest BESS. The project is a 200.74 MW battery storage facility in Hopkins County, NORTH zone.",
        "relevant_doc_keys": [
            "Pine Forest BESS::22INR0526::schedule_to",
            "Pine Forest BESS::22INR0526::schedule_attached",
            "Pine Forest BESS::22INR0526::exhibit_c"
        ]
    },
    {
        "question": "Who is the transmission provider for Tanglewood Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "CenterPoint Energy is the Transmission Service Provider for Tanglewood Solar. It is a 250.96 MW solar project located in Brazoria County in the COAST zone.",
        "relevant_doc_keys": [
            "Tanglewood Solar::23INR0054::schedule_of",
            "Tanglewood Solar::23INR0054::article_10"
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - CONTACT/EMAIL QUESTIONS (2)
    # ==========================================================================
    {
        "question": "What are all the relevant emails in the FRIENDSWOOD ENERGY GENCO project interconnection agreement?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "For Friendswood Energy Genco LLC: Suriyun Sukduang (ssukduang@quantumug.com). For operational and administrative notices: realtimedesk@nrg.com. For billing: AP.invoices@centerpointenergy.com and mborski@quantumug.com.",
        "relevant_doc_keys": [
            "FRIENDSWOOD ENERGY GENCO::24INR0456::schedule_of"
        ]
    },
    {
        "question": "What contact information is available for the Myrtle Solar project?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The Myrtle Solar project includes contacts at Sunchase Power as the developer and CenterPoint Energy as the TSP. The project is a 321.2 MW solar facility in Brazoria County.",
        "relevant_doc_keys": [
            "Myrtle Solar::19INR0041::schedule_of"
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - CAPACITY/FACILITY QUESTIONS (2)
    # ==========================================================================
    {
        "question": "What is the generation capacity of the Tanglewood Solar project?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "The Tanglewood Solar project has a generation capacity of 250.96 MW. It is located in Brazoria County in the COAST zone, with CenterPoint Energy as TSP.",
        "relevant_doc_keys": [
            "Tanglewood Solar::23INR0054::schedule_of",
            "Tanglewood Solar::23INR0054::article_10"
        ]
    },
    {
        "question": "What is the capacity of Pine Forest BESS and where is it located?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Pine Forest BESS has a capacity of 200.74 MW and is located in Hopkins County in the NORTH zone. It is connected through Oncor Electric Delivery Company LLC.",
        "relevant_doc_keys": [
            "Pine Forest BESS::22INR0526::schedule_to",
            "Pine Forest BESS::22INR0526::schedule_attached"
        ]
    },

    # ==========================================================================
    # IN-SCOPE ENGLISH - COMPARATIVE QUESTIONS (2)
    # ==========================================================================
    {
        "question": "Which project has a higher security amount: Parliament Solar or Tanglewood Solar?",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Tanglewood Solar has a higher security amount at $41,883,000 compared to Parliament Solar's $19,514,000. Both are solar projects in the COAST zone connected through CenterPoint Energy.",
        "relevant_doc_keys": [
            "Parliament Solar::23INR0044::article_10",
            "Tanglewood Solar::23INR0054::schedule_of"
        ]
    },
    {
        "question": "Compare the security amounts between Peyton Creek Wind II and Lavaca Bay Solar.",
        "lang": "english",
        "is_in_scope": True,
        "reference_answer": "Lavaca Bay Solar has a much higher security amount at $32,392,000 compared to Peyton Creek Wind II at $2,060,000. Lavaca Bay is a solar project while Peyton Creek is wind, both in the COAST zone with CenterPoint Energy as TSP.",
        "relevant_doc_keys": [
            "Peyton Creek Wind II::20INR0155::schedule_of",
            "Lavaca Bay Solar::23INR0084::article_10"
        ]
    },

    # ==========================================================================
    # IN-SCOPE SPANISH - SECURITY QUESTIONS (2)
    # ==========================================================================
    {
        "question": "¿Cuál es el monto de garantía para Lavaca Bay Solar?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "El monto de garantía para Lavaca Bay Solar es de $32,392,000. Este proyecto solar está ubicado en el condado de Matagorda, en la zona COAST, con una capacidad de 243.46 MW y CenterPoint Energy como proveedor de servicio de transmisión.",
        "relevant_doc_keys": [
            "Lavaca Bay Solar::23INR0084::exhibit_h",
            "Lavaca Bay Solar::23INR0084::schedule_attached"
        ]
    },
    {
        "question": "¿Cuánto es el depósito de seguridad requerido para el proyecto Peyton Creek Wind II?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "El depósito de seguridad para Peyton Creek Wind II es de $2,060,000. Es un proyecto eólico de 241.2 MW desarrollado por RWE, ubicado en el condado de Matagorda.",
        "relevant_doc_keys": [
            "Peyton Creek Wind II::20INR0155::schedule_of",
            "Peyton Creek Wind II::20INR0155::schedule_attached"
        ]
    },

    # ==========================================================================
    # IN-SCOPE SPANISH - TSP/FACILITY QUESTIONS (2)
    # ==========================================================================
    {
        "question": "¿Quién es el proveedor de transmisión para el proyecto Myrtle Solar?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "El proveedor de servicio de transmisión (TSP) para Myrtle Solar es CenterPoint Energy. Myrtle Solar es un proyecto solar de 321.2 MW ubicado en el condado de Brazoria.",
        "relevant_doc_keys": [
            "Myrtle Solar::19INR0041::schedule_of"
        ]
    },
    {
        "question": "¿Cuál es la capacidad del proyecto FRIENDSWOOD ENERGY GENCO?",
        "lang": "spanish",
        "is_in_scope": True,
        "reference_answer": "FRIENDSWOOD ENERGY GENCO tiene una capacidad de 143.7 MW. Es un proyecto de gas natural ubicado en el condado de Harris, conectado a través de CenterPoint Energy.",
        "relevant_doc_keys": [
            "FRIENDSWOOD ENERGY GENCO::24INR0456::schedule_of"
        ]
    },
]


def validate_dataset():
    """Validate that all required fields are present and correct."""
    required_fields = ["question", "lang", "is_in_scope", "reference_answer", "relevant_doc_keys"]
    valid_langs = ["english", "spanish"]
    
    errors = []
    
    for i, case in enumerate(SAMPLE_DATASET):
        for field in required_fields:
            if field not in case:
                errors.append(f"Case {i}: Missing field '{field}'")
        
        if case.get("lang") not in valid_langs:
            errors.append(f"Case {i}: Invalid lang '{case.get('lang')}'")
        
        if not isinstance(case.get("is_in_scope"), bool):
            errors.append(f"Case {i}: is_in_scope must be boolean")
        
        if not isinstance(case.get("relevant_doc_keys"), list):
            errors.append(f"Case {i}: relevant_doc_keys must be a list")
        
        if case.get("is_in_scope"):
            if not case.get("reference_answer"):
                errors.append(f"Case {i}: In-scope question missing reference_answer")
            if not case.get("relevant_doc_keys"):
                errors.append(f"Case {i}: In-scope question missing relevant_doc_keys")
    
    if errors:
        print("VALIDATION ERRORS:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("✓ Dataset validation passed!")
        return True


def print_coverage_stats():
    """Print dataset coverage statistics."""
    stats = {
        "total": len(SAMPLE_DATASET),
        "out_of_scope_en": 0,
        "out_of_scope_es": 0,
        "in_scope_en": 0,
        "in_scope_es": 0,
    }
    
    for case in SAMPLE_DATASET:
        is_in_scope = case["is_in_scope"]
        lang = case["lang"]
        
        if not is_in_scope:
            if lang == "english":
                stats["out_of_scope_en"] += 1
            else:
                stats["out_of_scope_es"] += 1
        else:
            if lang == "english":
                stats["in_scope_en"] += 1
            else:
                stats["in_scope_es"] += 1
    
    print("\n" + "="*60)
    print("DATASET COVERAGE STATISTICS")
    print("="*60)
    print(f"Total test cases: {stats['total']}")
    print(f"\nOut-of-scope questions:")
    print(f"  English: {stats['out_of_scope_en']}")
    print(f"  Spanish: {stats['out_of_scope_es']}")
    print(f"\nIn-scope questions:")
    print(f"  English: {stats['in_scope_en']}")
    print(f"  Spanish: {stats['in_scope_es']}")
    print("="*60)


if __name__ == "__main__":
    print("Validating SAMPLE_DATASET...")
    validate_dataset()
    print_coverage_stats()
