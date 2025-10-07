"""
HybridBail: Constants
Bail categories, patterns, weights, and thresholds
"""

# 17 Bail Categories
BAIL_CATEGORIES = {
    "anticipatory_bail": {"name": "Anticipatory Bail", "description": "Pre-arrest bail"},
    "regular_bail": {"name": "Regular Bail", "description": "Post-arrest bail"},
    "interim_bail": {"name": "Interim Bail", "description": "Temporary bail"},
    "default_bail": {"name": "Default Bail", "description": "Statutory bail"},
    "bail_cancellation": {"name": "Bail Cancellation", "description": "Revocation of bail"},
    "bail_conditions": {"name": "Bail with Conditions", "description": "Conditional bail"},
    "ndps_bail": {"name": "NDPS Bail", "description": "Narcotics cases"},
    "pocso_bail": {"name": "POCSO Bail", "description": "Child protection cases"},
    "uapa_bail": {"name": "UAPA Bail", "description": "Anti-terror cases"},
    "sc_st_act_bail": {"name": "SC/ST Act Bail", "description": "SC/ST Act cases"},
    "high_court_bail": {"name": "High Court Bail", "description": "High Court jurisdiction"},
    "supreme_court_bail": {"name": "Supreme Court Bail", "description": "Supreme Court"},
    "judicial_discretion": {"name": "Judicial Discretion", "description": "Discretionary bail"},
    "bail_precedent": {"name": "Bail Precedent", "description": "Landmark cases"},
}

# Decision Thresholds
DECISION_THRESHOLDS = {
    "auto_grant": 0.85,
    "auto_deny": 0.15,
    "human_intervention": (0.15, 0.85),
    "critical_category_threshold": 0.90
}

# Attribute Weights
ATTRIBUTE_WEIGHTS = {
    "default": {
        "crime_category": 0.30,
        "legal_sections": 0.25,
        "age_group": 0.10,
        "gender": 0.05,
        "region": 0.10,
        "criminal_history": 0.15,
        "custody_duration": 0.05
    }
}

# Human Intervention Triggers
HUMAN_INTERVENTION_TRIGGERS = {
    "low_confidence": 0.60,
    "critical_categories": ['ndps_bail', 'pocso_bail', 'uapa_bail'],
    "special_circumstances": ['Medical emergency', 'Pregnant woman', 'Juvenile'],
    "high_severity_crimes": ['302', '307', '376']
}

# Common Bail Conditions
COMMON_BAIL_CONDITIONS = [
    "Furnish personal bond and surety",
    "Surrender passport",
    "Not leave the country",
    "Appear before investigating officer",
    "Not tamper with evidence"
]

CATEGORY_SPECIFIC_CONDITIONS = {
    "ndps_bail": ["Regular drug testing", "Report to police station weekly"],
    "pocso_bail": ["No contact with victim", "Maintain distance from victim"],
}

# Supported Languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi"
}
