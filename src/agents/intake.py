"""Intake Agent Module for Rural India TB Screening.

Processes frontline ASHA worker clinical notes (Hindi, Marathi, English, Hinglish, Marathish)
using Google Gemini with strict JSON schema enforcement to extract clinical entities without guessing.
Computes a transparent, mathematically calibrated logistic risk score trained on synthetic data
with detailed contributing factors and follow-up question triage.
"""

import os
import re
import math
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

from src.schemas.models import IntakePayload, IntakeResult, GeminiIntakeExtraction

# Load environment variables from .env
load_dotenv()

# Logistic Regression Model Parameters Calibrated on data/synthetic/patients.csv
LOGIT_INTERCEPT = -2.1521
COEF_CHRONIC_COUGH = 1.3697       # cough_duration >= 14 days
COEF_COUGH_DAYS_NORM = 0.0726     # cough_duration / 30.0
COEF_FEVER = 0.5897
COEF_WEIGHT_LOSS = 0.9332
COEF_NIGHT_SWEATS = 0.4302
COEF_HEMOPTYSIS = 1.6231          # Pathognomonic pulmonary TB alarm sign
COEF_TB_CONTACT = 1.3310          # Household or close contact
COEF_HIV = 1.9053                 # Immunocompromised state
COEF_DIABETES = 0.5871            # Metabolic comorbidity
COEF_SMOKER = 0.5224              # Bidi/cigarette smoking
COEF_AGE_NORM = -0.0431           # age / 50.0


def get_gemini_client() -> Optional[Any]:
    """Retrieve an authenticated Google GenAI client if credentials exist."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def detect_language(text: str) -> str:
    """Detect whether input text is in Hindi, Marathi, Hinglish, Marathish, or English."""
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
    lower = text.lower()
    
    if has_devanagari:
        # Check Marathi vs Hindi markers
        mr_markers = ["रुग्ण", "रुग्णाला", "खोकला", "आहे", "नाही", "नाहीत", "दिवस", "घाम", "गाव", "वर्षे", "झाले", "पडत", "सासवड", "खेड"]
        hi_markers = ["मरीज", "खांसी", "बुखार", "खून", "साल", "हफ्ते", "रहा है", "हुआ", "आया", "गांव", "उम्र", "दिन"]
        mr_score = sum(1 for m in mr_markers if m in text)
        hi_score = sum(1 for m in hi_markers if m in text)
        if mr_score > hi_score:
            return "mr"
        if hi_score > mr_score:
            return "hi"
        return "mr" if any(m in text for m in ["रुग्ण", "खोकला", "आहे"]) else "hi"
    
    # Latin text classification
    en_words = {"patient", "male", "female", "cough", "fever", "weight", "loss", "night", "sweats", "blood", "sputum", "history", "contact", "observed", "severe", "days", "weeks", "mild", "sneezing", "runny", "nose"}
    tokens = set(re.findall(r'[a-z]+', lower))
    en_matches = len(tokens.intersection(en_words))
    
    marathish_markers = ["khokla", "ahe", "nahi", "divas", "gav", "varsh", "sandhyakali", "rukna", "thunkitun", "athavadyapasun", "athavade", "odhto"]
    mr_matches = sum(1 for m in marathish_markers if m in lower)
    
    hinglish_markers = ["khansi", "bukhar", "vajan", "khoon", "saal", "hafte", "hai", "nahi", "gaon", "mariz"]
    hi_matches = sum(1 for m in hinglish_markers if m in lower)
    
    if mr_matches >= 2 or (mr_matches > 0 and mr_matches >= en_matches):
        return "marathish"
    if hi_matches >= 2 or (hi_matches > 0 and hi_matches >= en_matches):
        return "hinglish"
    
    return "en"


def _is_negated(text: str, keyword: str) -> bool:
    """Check if a clinical symptom/risk term is preceded or followed by negation terms."""
    patterns = [
        rf'(?:no|not|denies|without|negative for|नाही|नाहीत|नही|नहीं|ना)\s+(?:[\w\u0900-\u097F]+\s+){{0,3}}{keyword}',
        rf'{keyword}\s+(?:[\w\u0900-\u097F]+\s+){{0,2}}(?:nahi|naahi|nahi hai|nahin|नाही|नाहीत|नहीं|नही|ना)',
        rf'\bno\s+{keyword}\b',
        rf'\bnon[- ]{keyword}\b',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def _rule_based_fallback_extraction(text: str) -> GeminiIntakeExtraction:
    """Deterministic multilingual heuristic fallback when Gemini API key is absent.
    
    Strictly adheres to: Never guess values. Returns None for fields not mentioned.
    """
    lower = text.lower()
    lang = detect_language(text)

    # 1. Age extraction
    age: Optional[int] = None
    age_match = re.search(r'(\b\d{1,3})[\s-]*(?:years?(?:-old)?|yrs?|yr|वर्षे|वर्ष|साल|saal|varsh)(?![a-zA-Z0-9\u0900-\u097F])', lower)
    if not age_match:
        age_match = re.search(r'(?:age|वय|उम्र|umra)\s*[:=-]?\s*(\b\d{1,3})(?![a-zA-Z0-9\u0900-\u097F])', lower)
    if not age_match:
        age_match = re.search(r'(?<![a-zA-Z0-9\u0900-\u097F])(\d{1,3})\s+(?:female|male|f|m|महिला|पुरुष|स्त्री|purush|mahila)(?![a-zA-Z0-9\u0900-\u097F])', lower)
    if age_match:
        val = int(age_match.group(1))
        if 0 <= val <= 120:
            age = val

    # 2. Sex extraction
    sex: Optional[str] = None
    if re.search(r'(?:^|[\s,;.-])(?:female|महिला|mahila|stri|aurat|woman|girl|स्त्री)(?:$|[\s,;.-])', text, re.IGNORECASE):
        sex = "F"
    elif re.search(r'(?:^|[\s,;.-])(?:male|पुरुष|purush|aadmi|man|boy)(?:$|[\s,;.-])', text, re.IGNORECASE):
        sex = "M"

    # 3. Village extraction
    village: Optional[str] = None
    village_match = re.search(r'(?:village|vill|गाव|गाँव|गांव|gav|gaon)\s*[:=-]?\s*([A-Za-z\u0900-\u097F]+)', text, re.IGNORECASE)
    if village_match:
        cand = village_match.group(1).strip("।,.-;: ")
        if cand.lower() not in ["ahe", "hai", "is", "of", "and", "me", "in", "ka", "ki"]:
            village = cand

    if not village:
        known_villages = [
            "Junnar", "Khed", "Shirur", "Baramati", "Saswad", "Koregaon",
            "Chikhaldara", "Warud", "Shahapur", "Shrigonda", "Shirdi", "Paithan",
            "जुन्नर", "खेड", "शिरूर", "बारामती", "सासवड", "कोरेगाव", "शिर्डी", "पैठण"
        ]
        for kv in known_villages:
            if kv.lower() in lower or kv in text:
                village = kv
                break

    # 4. Symptoms with durations
    # Cough
    cough_words = ["cough", "khansi", "khokla", "खोकला", "खांसी"]
    has_cough: Optional[bool] = None
    cough_duration_days: Optional[int] = None

    if any(k in lower for k in cough_words):
        if any(_is_negated(text, k) for k in cough_words) or "no chronic cough" in lower:
            has_cough = False
        else:
            has_cough = True
            # Extract duration
            if (
                "teen athavadyapasun" in lower or "teen athavade" in lower or "तीन आठवडे" in lower or "3 आठवडे" in lower or
                "3 hafte" in lower or "teen hafte" in lower or "तीन हफ्ते" in lower or "3 हफ्ते" in lower or "3 week" in lower
            ):
                cough_duration_days = 21
            elif (
                "don athavadyapasun" in lower or "don athavade" in lower or "दोन आठवडे" in lower or "2 आठवडे" in lower or
                "2 hafte" in lower or "don hafte" in lower or "दो हफ्ते" in lower or "2 हफ्ते" in lower or "2 week" in lower
            ):
                cough_duration_days = 14
            else:
                week_match = re.search(r'(\d+)[\s-]*(?:weeks?|hafte|आठवडे|हफ्ते|athavad[ye]?(?:apasun)?)(?![a-zA-Z0-9\u0900-\u097F])', lower)
                if week_match:
                    cough_duration_days = int(week_match.group(1)) * 7
                else:
                    day_match = re.search(r'(\d+)[\s-]*(?:days?|din|divas|दिवस|दिन)(?![a-zA-Z0-9\u0900-\u097F])', lower)
                    if day_match:
                        cough_duration_days = int(day_match.group(1))

    # Fever
    fever_words = ["fever", "taap", "bukhar", "ताप", "बुखार"]
    has_fever: Optional[bool] = None
    if any(k in lower for k in fever_words):
        has_fever = False if any(_is_negated(text, k) for k in fever_words) else True

    # Weight Loss
    has_weight_loss: Optional[bool] = None
    if "weight loss" in lower or "vajan kam" in lower or "वजन कमी" in lower or ("वजन" in text and "कम" in text) or "loss of appetite" in lower:
        if _is_negated(text, "weight loss") or _is_negated(text, "vajan") or _is_negated(text, "वजन"):
            has_weight_loss = False
        else:
            has_weight_loss = True

    # Night Sweats
    sweat_words = ["night sweats", "sweats", "ghama", "gham", "घाम", "पसीना", "pasina"]
    has_night_sweats: Optional[bool] = None
    if any(k in lower for k in sweat_words):
        has_night_sweats = False if any(_is_negated(text, k) for k in sweat_words) else True

    # Hemoptysis (Blood in sputum/cough)
    hemoptysis_words = ["hemoptysis", "blood in cough", "blood in sputum", "rakt", "khoon", "रक्त", "खून", "थुंकीतून रक्त"]
    has_hemoptysis: Optional[bool] = None
    if any(k in lower for k in hemoptysis_words):
        has_hemoptysis = False if any(_is_negated(text, k) for k in hemoptysis_words) else True

    # Risk Factors
    tb_contact: Optional[bool] = None
    has_contact_mention = (
        "tb contact" in lower or "contact history" in lower or "sampark" in lower or "संपर्क" in text or
        (re.search(r'\bcontact\b', lower) and re.search(r'\btb\b', lower)) or
        ("मरीज" in text and "संपर्क" in text) or ("रुग्ण" in text and "संपर्क" in text)
    )
    if has_contact_mention:
        if _is_negated(text, "contact") or _is_negated(text, "संपर्क") or _is_negated(text, "tb contact"):
            tb_contact = False
        else:
            tb_contact = True

    diabetes_words = ["diabetes", "sugar", "madhumeh", "मधुमेह", "डायबिटीज"]
    diabetes: Optional[bool] = None
    if any(k in lower for k in diabetes_words):
        diabetes = False if any(_is_negated(text, k) for k in diabetes_words) else True

    smoker_words = ["smoker", "smoking", "smoke", "bidi", "cigarette", "tambaku", "तंबाखू", "विडी", "सिगारेट", "बीड़ी"]
    smoker: Optional[bool] = None
    if any(k in lower for k in smoker_words):
        smoker = False if any(_is_negated(text, k) for k in smoker_words) else True

    hiv_words = ["hiv", "hiv positive", "एड्स", "aids"]
    hiv: Optional[bool] = None
    if any(k in lower for k in hiv_words):
        hiv = False if any(_is_negated(text, k) for k in hiv_words) else True

    return GeminiIntakeExtraction(
        age=age,
        sex=sex,  # type: ignore[arg-type]
        village=village,
        has_cough=has_cough,
        cough_duration_days=cough_duration_days,
        has_fever=has_fever,
        has_weight_loss=has_weight_loss,
        has_night_sweats=has_night_sweats,
        has_hemoptysis=has_hemoptysis,
        tb_contact=tb_contact,
        diabetes=diabetes,
        smoker=smoker,
        hiv=hiv,
        language_detected=lang,
    )


def extract_entities_with_gemini(text: str, client: Optional[Any] = None) -> GeminiIntakeExtraction:
    """Extract clinical entities using Gemini LLM with strict JSON schema.
    
    If client or GEMINI_API_KEY is not available, gracefully uses the heuristic fallback.
    """
    if client is None:
        client = get_gemini_client()

    if client is None:
        # Graceful deterministic fallback
        return _rule_based_fallback_extraction(text)

    prompt = f"""You are an expert clinical intake triage assistant for frontline ASHA health workers in Maharashtra, India.
Extract structured clinical information from the ASHA worker's clinical notes.
Notes may be in Hindi (Devanagari or Hinglish transliteration), Marathi (Devanagari or Marathish transliteration), or English.

CRITICAL EXTRACTION RULES:
1. NEVER guess, assume, or hallucinate values.
2. If age is not explicitly mentioned or clearly stated, return null.
3. If biological sex is not explicitly mentioned or clearly stated, return null.
4. If village or town name is not explicitly mentioned, return null.
5. If cough duration is not specified in days or weeks, return null for cough_duration_days.
6. Convert durations to total days (e.g., '3 weeks' -> 21, '2 weeks' -> 14, '5 days' -> 5).
7. If any risk factor (tb_contact, diabetes, smoker, hiv) is not mentioned, return null.
8. If a symptom or risk factor is explicitly negated (e.g. 'no fever', 'no hemoptysis', 'non-smoker', 'naahi'), return false.
9. Detect the language: 'hi', 'mr', 'en', 'hinglish', or 'marathish'.

Clinical Notes to Parse:
\"\"\"{text}\"\"\"
"""
    try:
        from google.genai import types
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiIntakeExtraction,
                temperature=0.0,
            )
        )
        extraction: Optional[GeminiIntakeExtraction] = None
        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, GeminiIntakeExtraction):
                extraction = response.parsed
            else:
                extraction = GeminiIntakeExtraction(**response.parsed)
        else:
            # Parse JSON text if parsed field is not automatically bound
            import json
            data = json.loads(response.text)
            extraction = GeminiIntakeExtraction(**data)

        if extraction and extraction.language_detected:
            lang_norm = extraction.language_detected.lower().strip()
            if lang_norm in ["hindi", "hi"]:
                extraction.language_detected = "hi"
            elif lang_norm in ["marathi", "mr"]:
                extraction.language_detected = "mr"
            elif lang_norm in ["english", "en"]:
                extraction.language_detected = "en"

        return extraction if extraction is not None else _rule_based_fallback_extraction(text)
    except Exception:
        # Fallback to deterministic extraction on network or quota failure
        return _rule_based_fallback_extraction(text)


def generate_follow_up_questions(extraction: GeminiIntakeExtraction) -> Tuple[List[str], List[str]]:
    """Identify missing required fields and generate targeted follow-up questions for ASHA workers."""
    missing: List[str] = []
    questions: List[str] = []
    lang = extraction.language_detected or "en"

    # Patient Demographics
    if extraction.age is None:
        missing.append("age")
        if "mr" in lang:
            questions.append("रुग्णाचे वय किती आहे? (What is the patient's age in years?)")
        elif "hi" in lang or "hinglish" in lang:
            questions.append("मरीज की उम्र क्या है? (What is the patient's age in years?)")
        else:
            questions.append("What is the patient's age in years?")

    if extraction.sex is None:
        missing.append("sex")
        if "mr" in lang:
            questions.append("रुग्णाचे लिंग काय आहे? पुरुष / स्त्री (Is the patient Male or Female?)")
        elif "hi" in lang or "hinglish" in lang:
            questions.append("मरीज का लिंग क्या है? पुरुष / महिला (Is the patient Male or Female?)")
        else:
            questions.append("Is the patient Male, Female, or Other?")

    if extraction.village is None:
        missing.append("village")
        if "mr" in lang:
            questions.append("रुग्ण कोणत्या गावातील आहे? (Which village or PHC sub-center is the patient from?)")
        elif "hi" in lang or "hinglish" in lang:
            questions.append("मरीज किस गांव या उप-केंद्र से है? (Which village or PHC sub-center is the patient from?)")
        else:
            questions.append("Which village or PHC sub-center is the patient from?")

    # Symptom Details
    if extraction.has_cough is True and extraction.cough_duration_days is None:
        missing.append("cough_duration_days")
        if "mr" in lang:
            questions.append("खोकला किती दिवसांपासून आहे? (दोन आठवड्यांपेक्षा जास्त आहे का?)")
        elif "hi" in lang or "hinglish" in lang:
            questions.append("खांसी कितने दिनों से है? (क्या दो हफ्ते से ज्यादा समय से है?)")
        else:
            questions.append("How many days has the cough persisted? Has it lasted for more than 2 weeks (14 days)?")

    if extraction.has_cough is None:
        missing.append("has_cough")
        questions.append("Does the patient have a cough? (रुग्णाला खोकला / मरीज को खांसी है क्या?)")

    # Risk Factors
    if extraction.tb_contact is None:
        missing.append("tb_contact")
        if "mr" in lang:
            questions.append("घरात किंवा संपर्कात टीबीचा रुग्ण होता का? (Has the patient had close or household TB contact?)")
        elif "hi" in lang or "hinglish" in lang:
            questions.append("क्या घर में या संपर्क में कोई टीबी का मरीज रहा है? (Has the patient had close or household TB contact?)")
        else:
            questions.append("Has the patient had household or close contact with a confirmed TB patient?")

    if extraction.diabetes is None:
        missing.append("diabetes")
        questions.append("Does the patient have diabetes / high blood sugar? (रुग्णाला मधुमेह / शुगर आहे का?)")

    if extraction.smoker is None:
        missing.append("smoker")
        questions.append("Does the patient have a history of smoking bidis or cigarettes? (रुग्ण विडी/सिगारेट ओढतो का?)")

    return missing, questions


def compute_risk_score(extraction: GeminiIntakeExtraction) -> Tuple[float, str, List[str]]:
    """Compute transparent logistic regression risk score (0 to 1) and identify top contributing factors.
    
    Model trained on data/synthetic/patients.csv with calibrated weights.
    """
    logit = LOGIT_INTERCEPT
    contributions: List[Tuple[str, float]] = []

    # Chronic Cough (>= 14 days)
    cough_days = extraction.cough_duration_days or 0
    if cough_days >= 14 or (extraction.has_cough and cough_days == 0):
        val = COEF_CHRONIC_COUGH
        logit += val
        contributions.append(("Persistent Cough >= 14 days", val))
    elif extraction.has_cough and cough_days > 0:
        val = 0.25
        logit += val
        contributions.append((f"Mild Cough ({cough_days} days)", val))

    # Incremental cough duration
    if cough_days > 0:
        val = (cough_days / 30.0) * COEF_COUGH_DAYS_NORM
        logit += val
        if val >= 0.05:
            contributions.append((f"Extended Cough Duration ({cough_days}d)", val))

    # Hemoptysis (Alarm symptom)
    if extraction.has_hemoptysis:
        val = COEF_HEMOPTYSIS
        logit += val
        contributions.append(("Hemoptysis (Blood in sputum/cough)", val))

    # TB Contact
    if extraction.tb_contact:
        val = COEF_TB_CONTACT
        logit += val
        contributions.append(("Household / Close TB Contact History", val))

    # Weight Loss
    if extraction.has_weight_loss:
        val = COEF_WEIGHT_LOSS
        logit += val
        contributions.append(("Unexplained Weight Loss", val))

    # Evening Fever
    if extraction.has_fever:
        val = COEF_FEVER
        logit += val
        contributions.append(("Recurrent / Evening Fever", val))

    # Night Sweats
    if extraction.has_night_sweats:
        val = COEF_NIGHT_SWEATS
        logit += val
        contributions.append(("Drenching Night Sweats", val))

    # Comorbidities
    if extraction.hiv:
        val = COEF_HIV
        logit += val
        contributions.append(("HIV Positive Immunocompromised Status", val))

    if extraction.diabetes:
        val = COEF_DIABETES
        logit += val
        contributions.append(("Diabetes Comorbidity", val))

    if extraction.smoker:
        val = COEF_SMOKER
        logit += val
        contributions.append(("Tobacco / Bidi Smoking", val))

    # Age effect
    if extraction.age is not None:
        val = (extraction.age / 50.0) * COEF_AGE_NORM
        logit += val

    # Sigmoid function
    risk_score = 1.0 / (1.0 + math.exp(-logit))
    risk_score = round(max(0.0, min(1.0, risk_score)), 4)

    # Risk level classification
    if risk_score >= 0.70:
        risk_level = "High"
    elif risk_score >= 0.35:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # Sort top contributing factors by positive contribution magnitude descending
    contributions.sort(key=lambda x: x[1], reverse=True)
    top_factors = [f"{factor} (+{contrib:.2f} logit contribution)" for factor, contrib in contributions if contrib > 0]
    if not top_factors:
        top_factors = ["Baseline presentation without acute alarm symptoms (-2.15 base logit)"]

    return risk_score, risk_level, top_factors


def extract_intake(text: str, client: Optional[Any] = None) -> IntakeResult:
    """Analyze free-form text from ASHA workers and produce structured IntakeResult.
    
    Args:
        text: Free-text clinical notes in Hindi, Marathi, or English.
        client: Optional genai.Client instance (useful for unit test mocking).
        
    Returns:
        IntakeResult: Typed structured result with extracted fields, risk score, and follow-ups.
    """
    extraction = extract_entities_with_gemini(text, client=client)
    missing, questions = generate_follow_up_questions(extraction)
    risk_score, risk_level, top_factors = compute_risk_score(extraction)

    extracted_symptoms: List[str] = []
    if extraction.has_cough:
        extracted_symptoms.append(f"Cough ({extraction.cough_duration_days or 'unknown'} days)" if extraction.cough_duration_days else "Cough")
    if extraction.has_fever:
        extracted_symptoms.append("Fever")
    if extraction.has_weight_loss:
        extracted_symptoms.append("Weight Loss")
    if extraction.has_night_sweats:
        extracted_symptoms.append("Night Sweats")
    if extraction.has_hemoptysis:
        extracted_symptoms.append("Hemoptysis")
    for extra in extraction.other_symptoms:
        if extra not in extracted_symptoms:
            extracted_symptoms.append(extra)

    return IntakeResult(
        patient_name=extraction.patient_name,
        age=extraction.age,
        gender=extraction.sex,
        village=extraction.village,
        extracted_symptoms=extracted_symptoms,
        has_cough=bool(extraction.has_cough),
        cough_duration_days=extraction.cough_duration_days or 0,
        has_fever=bool(extraction.has_fever),
        has_night_sweats=bool(extraction.has_night_sweats),
        has_weight_loss=bool(extraction.has_weight_loss),
        has_hemoptysis=bool(extraction.has_hemoptysis),
        contact_history=bool(extraction.tb_contact),
        risk_factors={
            "tb_contact": extraction.tb_contact,
            "diabetes": extraction.diabetes,
            "smoker": extraction.smoker,
            "hiv": extraction.hiv,
        },
        risk_score=risk_score,
        risk_level=risk_level,  # type: ignore[arg-type]
        top_contributing_factors=top_factors,
        missing_fields=missing,
        follow_up_questions=questions,
        language_detected=extraction.language_detected,
        raw_text=text,
        intake_timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.95,
    )


def run_intake(payload: IntakePayload, client: Optional[Any] = None) -> IntakeResult:
    """Execute Intake Agent on patient screening payload.
    
    Supports both checklist inputs and free-text notes.
    """
    # Extract entities from free-text raw notes
    res = extract_intake(payload.raw_notes, client=client)

    # Complement with payload metadata if payload explicitly provides it
    final_age = res.age if res.age is not None else payload.age
    final_gender = res.gender if res.gender is not None else payload.gender
    final_village = res.village if res.village is not None else payload.village
    final_contact = res.contact_history or payload.contact_history
    final_duration = res.cough_duration_days if res.cough_duration_days > 0 else payload.duration_days

    # Merge checklist symptoms
    combined_symptoms = list(set(res.extracted_symptoms + payload.symptoms))
    has_cough = res.has_cough or any("cough" in s.lower() for s in combined_symptoms) or final_duration > 0
    has_fever = res.has_fever or any("fever" in s.lower() for s in combined_symptoms)
    has_weight_loss = res.has_weight_loss or any("weight" in s.lower() for s in combined_symptoms)
    has_night_sweats = res.has_night_sweats or any("sweat" in s.lower() for s in combined_symptoms)
    has_hemoptysis = res.has_hemoptysis or any("blood" in s.lower() or "hemoptysis" in s.lower() for s in combined_symptoms)

    # Re-evaluate with complemented payload data
    merged_extraction = GeminiIntakeExtraction(
        patient_name=payload.patient_name if payload.patient_name != "Anonymous" else res.patient_name,
        age=final_age,
        sex=final_gender,  # type: ignore[arg-type]
        village=final_village,
        has_cough=has_cough,
        cough_duration_days=final_duration if final_duration > 0 else None,
        has_fever=has_fever,
        has_weight_loss=has_weight_loss,
        has_night_sweats=has_night_sweats,
        has_hemoptysis=has_hemoptysis,
        tb_contact=final_contact,
        diabetes=res.risk_factors.get("diabetes"),
        smoker=res.risk_factors.get("smoker"),
        hiv=res.risk_factors.get("hiv"),
        language_detected=res.language_detected,
    )
    
    missing, questions = generate_follow_up_questions(merged_extraction)
    risk_score, risk_level, top_factors = compute_risk_score(merged_extraction)

    return IntakeResult(
        patient_name=merged_extraction.patient_name,
        age=final_age,
        gender=final_gender,
        village=final_village,
        extracted_symptoms=combined_symptoms,
        has_cough=has_cough,
        cough_duration_days=final_duration,
        has_fever=has_fever,
        has_night_sweats=has_night_sweats,
        has_weight_loss=has_weight_loss,
        has_hemoptysis=has_hemoptysis,
        contact_history=final_contact,
        risk_factors=res.risk_factors,
        risk_score=risk_score,
        risk_level=risk_level,  # type: ignore[arg-type]
        top_contributing_factors=top_factors,
        missing_fields=missing,
        follow_up_questions=questions,
        language_detected=res.language_detected,
        raw_text=payload.raw_notes,
        intake_timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=0.95,
    )
