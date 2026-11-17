"""Synthetic Data Generator for Rural India TB Screening System.

Produces 5 correlated, realistic, synthetic datasets in data/synthetic/:
1. patients.csv: 3000 patients with demographics, Maharashtra districts/villages,
   symptoms, risk factors, and a probabilistic TB label.
2. lab_results.csv: CBNAAT molecular & sputum smear microscopy results with realistic noise.
3. treatment_logs.csv: 6-month daily adherence logs for TB-positive cases with dropout dynamics.
4. contacts.csv: Household contacts traced per confirmed index case.
5. drug_stock.csv: Monthly drug and cartridge stock & consumption per district.

RESEARCH PROTOTYPE DISCLAIMER:
All generated data is strictly synthetic for software prototyping and benchmarking.
No real patient or protected health information (PHI) is contained herein.
"""

import csv
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

# Ensure workspace root is available
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Deterministic random seed
DEFAULT_SEED = 42

# Geographic mapping: Real Maharashtra districts and rural primary healthcare villages/blocks
MAHARASHTRA_GEOGRAPHY = {
    "Pune": ["Junnar", "Shirur", "Khed", "Ambegaon", "Baramati", "Indapur", "Bhor", "Velhe"],
    "Ahmednagar": ["Parner", "Sangamner", "Rahata", "Akole", "Shevgaon", "Shrigonda", "Pathardi"],
    "Satara": ["Karad", "Patan", "Wai", "Koregaon", "Jawali", "Khandala", "Phaltan"],
    "Nashik": ["Dindori", "Igatpuri", "Trimbakeshwar", "Kalwan", "Surgana", "Peint", "Malegaon"],
    "Solapur": ["Barshi", "Pandharpur", "Madha", "Karmala", "Mohol", "Mangalwedha", "Sangola"],
    "Kolhapur": ["Shahuwadi", "Radhanagari", "Bhudargad", "Ajara", "Panhala", "Gaganbawda"],
    "Thane": ["Murbad", "Shahapur", "Bhiwandi Rural"],
    "Nanded": ["Kinwat", "Mahur", "Hadgaon", "Mukhed", "Degloor", "Bhokar"],
    "Amravati": ["Dharni (Melghat)", "Chikhaldara", "Achalpur", "Morshi", "Warud"],
    "Nagpur": ["Katol", "Narkhed", "Ramtek", "Parseoni", "Umred", "Kuhi"],
}

INDIAN_FIRST_NAMES_M = [
    "Ramesh", "Suresh", "Santosh", "Ganesh", "Sachin", "Prakash", "Sunil", "Anil", "Deepak",
    "Vijay", "Sanjay", "Dattatray", "Pandurang", "Tukaram", "Shivaji", "Vikas", "Ashok",
    "Vitthal", "Baban", "Shankar", "Maruti", "Nana", "Popat", "Balasaheb", "Eknath"
]

INDIAN_FIRST_NAMES_F = [
    "Sunita", "Anita", "Sangeeta", "Shobha", "Mangal", "Rekha", "Usha", "Surekha", "Kavita",
    "Pooja", "Vaishali", "Manisha", "Lata", "Suvarna", "Chhaya", "Rukmini", "Parvati",
    "Kamal", "Sindhu", "Savita", "Asha", "Meena", "Vandana", "Geeta", "Jyoti"
]

MAHARASHTRIAN_SURNAMES = [
    "Patil", "Deshmukh", "Jadhav", "Shinde", "Pawar", "Kulkarni", "Chavan", "Gaekwad",
    "More", "Kadam", "Bhosale", "Kale", "Gaikwad", "Tambe", "Wagh", "Gite", "Kharat",
    "Gholap", "Sawant", "Salunkhe", "Thorat", "Mane", "Jagatap", "Sonawane", "Ghogare"
]


def generate_patients(num_patients: int = 3000, seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """Generate 3000 synthetic patient records with realistic epidemiological features."""
    random.seed(seed)
    districts = list(MAHARASHTRA_GEOGRAPHY.keys())
    patients = []

    for i in range(1, num_patients + 1):
        patient_id = f"PAT-{i:04d}"
        sex = "M" if random.random() < 0.54 else "F"

        # Demographics
        if sex == "M":
            first_name = random.choice(INDIAN_FIRST_NAMES_M)
        else:
            first_name = random.choice(INDIAN_FIRST_NAMES_F)
        surname = random.choice(MAHARASHTRIAN_SURNAMES)
        patient_name = f"{first_name} {surname}"

        # Age distribution: Bimodal rural adult population (mean ~42, std ~16)
        age = int(random.gauss(42, 16))
        age = max(15, min(85, age))

        district = random.choice(districts)
        village = random.choice(MAHARASHTRA_GEOGRAPHY[district])

        # Distance to PHC / DMC: log-normal distribution (median ~6 km, tail up to 35 km)
        distance_km = round(random.lognormvariate(1.7, 0.7), 1)
        distance_km = max(0.8, min(42.0, distance_km))

        # Risk Factors
        diabetes = random.random() < 0.13  # 13% prevalence in rural adults
        hiv_status = "Positive" if random.random() < 0.02 else "Negative"  # 2% HIV prevalence
        if sex == "M":
            smoker = random.random() < 0.28  # Bidi/cigarette smoking in adult males
        else:
            smoker = random.random() < 0.04
        tb_contact = random.random() < 0.15  # Household or close contact

        # Symptoms
        # Cough duration in days: mix of acute and chronic
        if random.random() < 0.65:
            # Significant cough suspect
            cough_duration = int(random.expovariate(1.0 / 18.0) + 7)
        else:
            # Mild or transient cough
            cough_duration = random.randint(0, 10)
        cough_duration = min(cough_duration, 90)

        has_cough = cough_duration > 0
        has_fever = random.random() < (0.60 if cough_duration >= 14 else 0.25)
        has_weight_loss = random.random() < (0.50 if cough_duration >= 14 else 0.15)
        has_night_sweats = random.random() < (0.40 if cough_duration >= 14 else 0.10)
        has_hemoptysis = random.random() < (0.12 if cough_duration >= 14 else 0.01)

        # Probabilistic rule for true TB positive status (calibrated logit model)
        # Target screening positivity in rural health camps: ~18% - 22%
        logit = -3.70
        if cough_duration >= 14:
            logit += 1.70
        elif cough_duration > 0:
            logit += 0.30

        if has_fever:
            logit += 0.70
        if has_weight_loss:
            logit += 0.90
        if has_night_sweats:
            logit += 0.65
        if has_hemoptysis:
            logit += 1.50
        if tb_contact:
            logit += 1.25
        if hiv_status == "Positive":
            logit += 1.80
        if diabetes:
            logit += 0.50
        if smoker:
            logit += 0.40

        prob_tb = 1.0 / (1.0 + math.exp(-logit))
        tb_positive = 1 if random.random() < prob_tb else 0

        patients.append({
            "patient_id": patient_id,
            "patient_name": patient_name,
            "age": age,
            "sex": sex,
            "district": district,
            "village": village,
            "distance_to_phc_km": distance_km,
            "cough_duration_days": cough_duration,
            "has_cough": int(has_cough),
            "has_fever": int(has_fever),
            "has_weight_loss": int(has_weight_loss),
            "has_night_sweats": int(has_night_sweats),
            "has_hemoptysis": int(has_hemoptysis),
            "diabetes": int(diabetes),
            "hiv_status": hiv_status,
            "smoker": int(smoker),
            "tb_contact": int(tb_contact),
            "tb_positive": tb_positive,
        })

    return patients


def generate_lab_results(patients: List[Dict[str, Any]], seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """Generate molecular (CBNAAT) and sputum smear microscopy results correlated with TB label."""
    random.seed(seed + 10)
    lab_records = []
    lab_counter = 1
    base_date = datetime(2025, 1, 15, tzinfo=timezone.utc)

    for p in patients:
        is_tb = p["tb_positive"] == 1
        # Test collection date within random window
        test_offset = random.randint(1, 300)
        collect_date = (base_date + timedelta(days=test_offset)).strftime("%Y-%m-%d")
        result_date = (base_date + timedelta(days=test_offset + 1)).strftime("%Y-%m-%d")

        # 1. CBNAAT / GeneXpert Test (Sensitivity ~91%, Specificity ~98.5%)
        # All patients with cough >= 14 days or HIV or high risk receive CBNAAT upfront per NTEP
        warrants_cbnaat = p["cough_duration_days"] >= 14 or p["hiv_status"] == "Positive" or is_tb or (random.random() < 0.40)
        if warrants_cbnaat:
            if is_tb:
                cbnaat_detected = random.random() < 0.91  # 91% sensitivity
            else:
                cbnaat_detected = random.random() < 0.015  # 98.5% specificity

            if cbnaat_detected:
                result_str = "MTB Detected"
                burden = random.choice(["Low", "Medium", "High", "Very Low"])
                # Rifampicin resistance detected in ~4% of positives (modeling MDR-TB)
                rif_res = "Detected" if random.random() < 0.04 else "Not Detected"
            else:
                result_str = "MTB Not Detected"
                burden = "None"
                rif_res = "Not Applicable"

            lab_records.append({
                "lab_id": f"LAB-{lab_counter:06d}",
                "patient_id": p["patient_id"],
                "test_type": "CBNAAT / GeneXpert",
                "sample_type": "Sputum",
                "collection_date": collect_date,
                "result_date": result_date,
                "facility": f"{p['district']} District TB Centre",
                "result": result_str,
                "mtb_burden": burden,
                "rifampicin_resistance": rif_res,
                "concordant_with_true_label": int(cbnaat_detected == is_tb),
            })
            lab_counter += 1

        # 2. Sputum Smear Microscopy (Sensitivity ~65%, Specificity ~96%)
        # Evaluated at PHC Designated Microscopy Centre (DMC)
        warrants_smear = p["has_cough"] == 1 or (random.random() < 0.50)
        if warrants_smear:
            if is_tb:
                smear_positive = random.random() < 0.66
            else:
                smear_positive = random.random() < 0.035

            if smear_positive:
                grade = random.choice(["Scanty (1-9 AFB/100 fields)", "1+ (10-99 AFB/100 fields)", "2+ (1-10 AFB/field)", "3+ (>10 AFB/field)"])
                smear_res = "Positive (AFB Seen)"
            else:
                grade = "Negative (No AFB Seen)"
                smear_res = "Negative"

            lab_records.append({
                "lab_id": f"LAB-{lab_counter:06d}",
                "patient_id": p["patient_id"],
                "test_type": "Sputum Smear Microscopy",
                "sample_type": "Sputum",
                "collection_date": collect_date,
                "result_date": result_date,
                "facility": f"{p['village']} DMC",
                "result": smear_res,
                "mtb_burden": grade,
                "rifampicin_resistance": "Not Applicable",
                "concordant_with_true_label": int(smear_positive == is_tb),
            })
            lab_counter += 1

    return lab_records


def generate_treatment_logs(patients: List[Dict[str, Any]], seed: int = DEFAULT_SEED) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generate 6-month (168 days) daily dose adherence logs for TB-positive patients.

    Models daily dose adherence and patient dropout dynamics driven by:
    - Distance to health post
    - Age
    - Cumulative and recent missed doses (interruption momentum)
    - Drug adverse side effects
    """
    random.seed(seed + 20)
    tb_positive_patients = [p for p in patients if p["tb_positive"] == 1]
    daily_logs = []
    log_counter = 1

    start_date = datetime(2025, 2, 1, tzinfo=timezone.utc)
    patient_outcomes: Dict[str, Dict[str, Any]] = {}

    for p in tb_positive_patients:
        pid = p["patient_id"]
        dist = p["distance_to_phc_km"]
        age = p["age"]

        # Individual adherence vulnerability profile:
        # Dropout hazard depends directly on: distance, age, side effects, and earlier missed doses.
        vulnerability = 0.018 + (dist / 110.0)
        if dist >= 12.0:
            vulnerability += 0.038
        if age >= 55:
            vulnerability += 0.032
        if random.random() < 0.14:  # Socioeconomic / migration barrier
            vulnerability += 0.065

        base_miss_prob = vulnerability

        consecutive_misses = 0
        total_missed = 0
        dropped_out = False
        dropout_day = None
        patient_records = []

        # 168 days = 24 weeks standard daily DOTS regimen
        for day in range(1, 169):
            phase = "Intensive Phase (4FDC)" if day <= 56 else "Continuation Phase (2FDC)"
            current_date = (start_date + timedelta(days=day)).strftime("%Y-%m-%d")

            if dropped_out:
                # Once dropped out, patient records 0 dose with Dropped Out status
                patient_records.append({
                    "log_id": f"TRT-{log_counter:07d}",
                    "patient_id": pid,
                    "day_number": day,
                    "date": current_date,
                    "regimen_phase": phase,
                    "dose_taken": 0,
                    "side_effects": "None",
                    "asha_verified": 0,
                    "status": "Dropped Out / Defaulted",
                })
                log_counter += 1
                continue

            # Model dynamic adverse side effects (more frequent in Intensive Phase days 7-42)
            has_side_effect = False
            side_effect_str = "None"
            if 7 <= day <= 42:
                if random.random() < 0.12:
                    has_side_effect = True
                    side_effect_str = random.choice(["Mild Nausea", "Gastritis", "Joint Pain", "Skin Rash", "Hepatitis/Jaundice"])
            elif day > 42 and random.random() < 0.04:
                has_side_effect = True
                side_effect_str = random.choice(["Joint Pain", "Mild Nausea", "Peripheral Neuropathy"])

            # Calculate today's miss probability driven by distance, age, side effects, and momentum
            today_miss_prob = base_miss_prob
            if has_side_effect:
                today_miss_prob += (0.30 if side_effect_str == "Hepatitis/Jaundice" else 0.16)
            if consecutive_misses > 0:
                # Earlier missed doses create non-adherence momentum
                today_miss_prob += min(0.60, consecutive_misses * 0.14)

            dose_taken = 1 if random.random() > today_miss_prob else 0

            if dose_taken == 0:
                consecutive_misses += 1
                total_missed += 1
            else:
                consecutive_misses = 0

            # ASHA home verification (higher when patient takes medicine)
            asha_verified = 1 if (dose_taken == 1 and random.random() < 0.88) else (1 if random.random() < 0.35 else 0)

            # NTEP Loss-to-Follow-up criteria:
            # Treatment interruption >= 14 consecutive missed days or >= 28 cumulative missed doses with active interruption (>=5 days)
            if consecutive_misses >= 14 or (day >= 56 and total_missed >= 28 and consecutive_misses >= 5):
                dropped_out = True
                dropout_day = day
                status = "Dropped Out / Defaulted"
            else:
                status = "Active"

            patient_records.append({
                "log_id": f"TRT-{log_counter:07d}",
                "patient_id": pid,
                "day_number": day,
                "date": current_date,
                "regimen_phase": phase,
                "dose_taken": dose_taken,
                "side_effects": side_effect_str,
                "asha_verified": asha_verified,
                "status": status,
            })
            log_counter += 1

        daily_logs.extend(patient_records)

        # Record patient summary metrics
        adherence_rate = round(((168 - total_missed) / 168) * 100, 1)
        patient_outcomes[pid] = {
            "dropped_out": dropped_out,
            "dropout_day": dropout_day,
            "total_missed_doses": total_missed,
            "adherence_rate_pct": adherence_rate,
            "final_status": "Dropped Out / Defaulted" if dropped_out else ("Completed Treatment" if adherence_rate >= 80.0 else "Poor Adherence - Needs Retreatment"),
        }

    return daily_logs, patient_outcomes


def generate_contacts(patients: List[Dict[str, Any]], seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """Generate household contacts traced per confirmed TB-positive case."""
    random.seed(seed + 30)
    tb_positive_patients = [p for p in patients if p["tb_positive"] == 1]
    contacts = []
    contact_counter = 1

    relationships = ["Spouse", "Child", "Parent", "Sibling", "Grandparent", "In-law"]

    for p in tb_positive_patients:
        # 2 to 5 household contacts per rural household
        num_contacts = random.randint(2, 5)

        for _ in range(num_contacts):
            rel = random.choice(relationships)
            if rel == "Child":
                contact_age = random.randint(1, 16)
                c_sex = "M" if random.random() < 0.52 else "F"
                c_name = f"{random.choice(INDIAN_FIRST_NAMES_M if c_sex == 'M' else INDIAN_FIRST_NAMES_F)} {p['patient_name'].split()[-1]}"
            elif rel == "Spouse":
                contact_age = max(18, p["age"] + random.randint(-4, 4))
                c_sex = "F" if p["sex"] == "M" else "M"
                c_name = f"{random.choice(INDIAN_FIRST_NAMES_F if c_sex == 'F' else INDIAN_FIRST_NAMES_M)} {p['patient_name'].split()[-1]}"
            elif rel in ["Parent", "Grandparent"]:
                contact_age = min(88, p["age"] + random.randint(18, 30))
                c_sex = "M" if random.random() < 0.50 else "F"
                c_name = f"{random.choice(INDIAN_FIRST_NAMES_M if c_sex == 'M' else INDIAN_FIRST_NAMES_F)} {p['patient_name'].split()[-1]}"
            else:
                contact_age = max(14, min(75, p["age"] + random.randint(-12, 12)))
                c_sex = "M" if random.random() < 0.50 else "F"
                c_name = f"{random.choice(INDIAN_FIRST_NAMES_M if c_sex == 'M' else INDIAN_FIRST_NAMES_F)} {p['patient_name'].split()[-1]}"

            # Clinical contact tracing evaluation
            symptomatic = random.random() < 0.16  # ~16% contacts symptomatic
            eval_tpt = random.random() < 0.92     # 92% evaluated for TPT (Preventive treatment)

            # Children under 5 or high-risk asymptomatic contacts prioritized for TPT
            if eval_tpt and not symptomatic:
                if contact_age <= 6 or random.random() < 0.45:
                    tpt_initiated = 1
                    status = "Initiated on TPT (3HP / 6H)"
                else:
                    tpt_initiated = 0
                    status = "Screened Negative - Under Surveillance"
            elif symptomatic:
                tpt_initiated = 0
                if random.random() < 0.28:
                    status = "Confirmed Secondary TB - Referred for ATT"
                else:
                    status = "Symptomatic - Testing Pending"
            else:
                tpt_initiated = 0
                status = "Screened Negative"

            contacts.append({
                "contact_id": f"CNT-{contact_counter:05d}",
                "index_patient_id": p["patient_id"],
                "index_patient_name": p["patient_name"],
                "contact_name": c_name,
                "age": contact_age,
                "sex": c_sex,
                "relationship": rel,
                "village": p["village"],
                "district": p["district"],
                "symptoms_present": int(symptomatic),
                "evaluated_for_tpt": int(eval_tpt),
                "tpt_initiated": tpt_initiated,
                "contact_outcome": status,
            })
            contact_counter += 1

    return contacts


def generate_drug_stock(seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """Generate monthly drug stock and consumption logs per Maharashtra district for 12 months."""
    random.seed(seed + 40)
    stock_records = []
    record_counter = 1

    districts = list(MAHARASHTRA_GEOGRAPHY.keys())
    months = [f"2025-{m:02d}" for m in range(1, 13)]

    drugs = [
        ("4FDC Adult (HRZE) Blister Packs", 3800, 1100),
        ("2FDC Adult (HR) Blister Packs", 5400, 1500),
        ("Pediatric FDC Blister Packs", 1200, 350),
        ("CBNAAT GeneXpert Cartridges", 850, 240),
        ("Pyridoxine (Vitamin B6) Tablets", 4200, 1200),
    ]

    for district in districts:
        # District size scalar (Pune/Thane/Nagpur have higher programmatic load)
        scale = 1.6 if district in ["Pune", "Thane", "Nagpur"] else 1.0

        for drug_name, base_stock, base_consumption in drugs:
            current_opening = int(base_stock * scale * random.uniform(0.9, 1.25))
            monthly_demand = int(base_consumption * scale)
            buffer_threshold = int(monthly_demand * 1.5)  # 1.5 months buffer stock requirement

            for m in months:
                # Consumption with slight seasonal variance (monsoon / winter peaks)
                seasonal_factor = random.uniform(0.85, 1.20)
                consumed = int(monthly_demand * seasonal_factor)

                # Damaged or expired
                damaged = int(random.uniform(2, 18))

                # Periodic state central medical store replenishment
                # Happens every 2-3 months or when stock drops
                if current_opening < buffer_threshold or random.random() < 0.40:
                    received = int(monthly_demand * random.uniform(1.8, 3.2))
                else:
                    received = 0

                closing = current_opening + received - consumed - damaged
                if closing < 0:
                    closing = 0

                if closing == 0:
                    alert = "Critical Stockout"
                elif closing < (buffer_threshold * 0.6):
                    alert = "Severe Low Stock"
                elif closing < buffer_threshold:
                    alert = "Buffer Stock Warning"
                else:
                    alert = "Adequate"

                stock_records.append({
                    "record_id": f"STK-{record_counter:05d}",
                    "district": district,
                    "month": m,
                    "item_name": drug_name,
                    "opening_stock": current_opening,
                    "received_stock": received,
                    "consumed_stock": consumed,
                    "damaged_or_expired": damaged,
                    "closing_stock": closing,
                    "buffer_threshold": buffer_threshold,
                    "stock_status": alert,
                })
                record_counter += 1

                # Next month opening is this month closing
                current_opening = closing

    return stock_records


def write_csv(filepath: str, data: List[Dict[str, Any]]):
    """Write list of dictionaries to CSV file."""
    if not data:
        return
    fieldnames = list(data[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def generate_all_datasets(output_dir: str = "data/synthetic", seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Generate all 5 synthetic datasets and output to disk."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating synthetic datasets (Seed: {seed}) in: {output_dir}")

    # 1. Patients
    patients = generate_patients(num_patients=3000, seed=seed)
    patients_file = os.path.join(output_dir, "patients.csv")
    write_csv(patients_file, patients)
    print(f"-> Generated {len(patients):,} patients in {patients_file}")

    # 2. Lab Results
    labs = generate_lab_results(patients, seed=seed)
    labs_file = os.path.join(output_dir, "lab_results.csv")
    write_csv(labs_file, labs)
    print(f"-> Generated {len(labs):,} lab records in {labs_file}")

    # 3. Treatment Logs
    treatment_logs, patient_outcomes = generate_treatment_logs(patients, seed=seed)
    treatment_file = os.path.join(output_dir, "treatment_logs.csv")
    write_csv(treatment_file, treatment_logs)
    print(f"-> Generated {len(treatment_logs):,} daily adherence logs in {treatment_file}")

    # 4. Contacts
    contacts = generate_contacts(patients, seed=seed)
    contacts_file = os.path.join(output_dir, "contacts.csv")
    write_csv(contacts_file, contacts)
    print(f"-> Generated {len(contacts):,} contact tracing records in {contacts_file}")

    # 5. Drug Stock
    stocks = generate_drug_stock(seed=seed)
    stocks_file = os.path.join(output_dir, "drug_stock.csv")
    write_csv(stocks_file, stocks)
    print(f"-> Generated {len(stocks):,} monthly drug stock records in {stocks_file}")

    # Summary Statistics Calculation
    stats = compute_summary_statistics(patients, labs, treatment_logs, patient_outcomes, contacts, stocks)
    return stats


def compute_summary_statistics(
    patients: List[Dict[str, Any]],
    labs: List[Dict[str, Any]],
    treatment_logs: List[Dict[str, Any]],
    patient_outcomes: Dict[str, Any],
    contacts: List[Dict[str, Any]],
    stocks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute comprehensive statistical metrics across all datasets."""
    n_patients = len(patients)
    tb_pos = sum(1 for p in patients if p["tb_positive"] == 1)
    positivity_rate = (tb_pos / n_patients) * 100

    male_count = sum(1 for p in patients if p["sex"] == "M")
    female_count = sum(1 for p in patients if p["sex"] == "F")
    avg_age = sum(p["age"] for p in patients) / n_patients
    avg_distance = sum(p["distance_to_phc_km"] for p in patients) / n_patients

    chronic_cough = sum(1 for p in patients if p["cough_duration_days"] >= 14)
    hemoptysis = sum(1 for p in patients if p["has_hemoptysis"] == 1)
    weight_loss = sum(1 for p in patients if p["has_weight_loss"] == 1)
    tb_contact_history = sum(1 for p in patients if p["tb_contact"] == 1)
    hiv_positive = sum(1 for p in patients if p["hiv_status"] == "Positive")

    # Lab Stats
    cbnaat_tests = [l for l in labs if l["test_type"] == "CBNAAT / GeneXpert"]
    cbnaat_pos = sum(1 for l in cbnaat_tests if l["result"] == "MTB Detected")
    cbnaat_concordance = (sum(l["concordant_with_true_label"] for l in cbnaat_tests) / len(cbnaat_tests)) * 100 if cbnaat_tests else 0.0

    smear_tests = [l for l in labs if l["test_type"] == "Sputum Smear Microscopy"]
    smear_pos = sum(1 for l in smear_tests if "Positive" in l["result"])
    smear_concordance = (sum(l["concordant_with_true_label"] for l in smear_tests) / len(smear_tests)) * 100 if smear_tests else 0.0

    # Treatment Outcomes
    n_tb_patients = len(patient_outcomes)
    dropouts = sum(1 for o in patient_outcomes.values() if o["dropped_out"])
    dropout_rate = (dropouts / n_tb_patients) * 100 if n_tb_patients else 0.0
    avg_adherence = sum(o["adherence_rate_pct"] for o in patient_outcomes.values()) / n_tb_patients if n_tb_patients else 0.0

    # Contacts Stats
    n_contacts = len(contacts)
    contacts_per_case = n_contacts / tb_pos if tb_pos else 0.0
    tpt_initiated_count = sum(1 for c in contacts if c["tpt_initiated"] == 1)
    secondary_tb_found = sum(1 for c in contacts if "Confirmed Secondary TB" in c["contact_outcome"])

    # Stock Stats
    stockout_events = sum(1 for s in stocks if s["stock_status"] in ["Critical Stockout", "Severe Low Stock"])

    return {
        "patients": {
            "total_patients": n_patients,
            "tb_positive_count": tb_pos,
            "tb_positivity_pct": round(positivity_rate, 2),
            "male_count": male_count,
            "female_count": female_count,
            "mean_age": round(avg_age, 1),
            "mean_distance_to_phc_km": round(avg_distance, 1),
            "chronic_cough_ge_14d": chronic_cough,
            "hemoptysis_count": hemoptysis,
            "weight_loss_count": weight_loss,
            "tb_contact_count": tb_contact_history,
            "hiv_positive_count": hiv_positive,
        },
        "labs": {
            "total_lab_tests": len(labs),
            "cbnaat_count": len(cbnaat_tests),
            "cbnaat_positive": cbnaat_pos,
            "cbnaat_concordance_pct": round(cbnaat_concordance, 2),
            "smear_count": len(smear_tests),
            "smear_positive": smear_pos,
            "smear_concordance_pct": round(smear_concordance, 2),
        },
        "treatment": {
            "total_tb_patients_treated": n_tb_patients,
            "total_daily_dose_logs": len(treatment_logs),
            "dropout_count": dropouts,
            "dropout_rate_pct": round(dropout_rate, 2),
            "mean_adherence_rate_pct": round(avg_adherence, 2),
        },
        "contacts": {
            "total_contacts_traced": n_contacts,
            "contacts_per_confirmed_case": round(contacts_per_case, 2),
            "tpt_initiated_count": tpt_initiated_count,
            "secondary_tb_detected_count": secondary_tb_found,
        },
        "drug_stock": {
            "total_stock_records": len(stocks),
            "districts_tracked": len(set(s["district"] for s in stocks)),
            "critical_or_low_stock_events": stockout_events,
        },
    }


def print_summary_report(stats: Dict[str, Any]):
    """Print clean formatted statistical report to console."""
    p = stats["patients"]
    l = stats["labs"]
    t = stats["treatment"]
    c = stats["contacts"]
    s = stats["drug_stock"]

    print("\n" + "=" * 80)
    print("                SYNTHETIC TB SCREENING DATASET - SUMMARY STATISTICS")
    print("=" * 80)
    print(f"DISCLAIMER: All data is synthetic for research prototyping and not clinically validated.\n")

    print(f"[1] PATIENTS DATASET (patients.csv)")
    print(f"  * Total Patients Screened   : {p['total_patients']:,}")
    print(f"  * Gender Breakdown          : Male={p['male_count']:,} ({p['male_count']/p['total_patients']*100:.1f}%), Female={p['female_count']:,} ({p['female_count']/p['total_patients']*100:.1f}%)")
    print(f"  * Mean Age                  : {p['mean_age']} years")
    print(f"  * Mean Distance to PHC      : {p['mean_distance_to_phc_km']} km")
    print(f"  * Chronic Cough (>= 14 days): {p['chronic_cough_ge_14d']:,} ({p['chronic_cough_ge_14d']/p['total_patients']*100:.1f}%)")
    print(f"  * Weight Loss Reported      : {p['weight_loss_count']:,} ({p['weight_loss_count']/p['total_patients']*100:.1f}%)")
    print(f"  * Hemoptysis (Blood in cough): {p['hemoptysis_count']:,} ({p['hemoptysis_count']/p['total_patients']*100:.1f}%)")
    print(f"  * Known TB Contact History  : {p['tb_contact_count']:,} ({p['tb_contact_count']/p['total_patients']*100:.1f}%)")
    print(f"  * HIV Positive              : {p['hiv_positive_count']:,} ({p['hiv_positive_count']/p['total_patients']*100:.1f}%)")
    print(f"  * TB Positive (Probabilistic): {p['tb_positive_count']:,} ({p['tb_positivity_pct']}%)")

    print(f"\n[2] LAB INVESTIGATIONS (lab_results.csv)")
    print(f"  * Total Lab Tests Conducted : {l['total_lab_tests']:,}")
    print(f"  * CBNAAT / GeneXpert Tests  : {l['cbnaat_count']:,} (Positive={l['cbnaat_positive']:,}, Concordance={l['cbnaat_concordance_pct']}%)")
    print(f"  * Sputum Smear Microscopy   : {l['smear_count']:,} (Positive={l['smear_positive']:,}, Concordance={l['smear_concordance_pct']}%)")

    print(f"\n[3] 6-MONTH TREATMENT ADHERENCE LOGS (treatment_logs.csv)")
    print(f"  * TB Positive Patients Tracked: {t['total_tb_patients_treated']:,}")
    print(f"  * Total Daily Dose Logs      : {t['total_daily_dose_logs']:,} days logged")
    print(f"  * Overall Mean Adherence     : {t['mean_adherence_rate_pct']}%")
    print(f"  * Defaulted / Dropped Out    : {t['dropout_count']:,} ({t['dropout_rate_pct']}%)")
    print(f"  * Completed or Maintained ATT: {t['total_tb_patients_treated'] - t['dropout_count']:,} ({100 - t['dropout_rate_pct']:.1f}%)")

    print(f"\n[4] CONTACT TRACING (contacts.csv)")
    print(f"  * Household Contacts Traced  : {c['total_contacts_traced']:,}")
    print(f"  * Mean Contacts per Case     : {c['contacts_per_confirmed_case']}")
    print(f"  * TPT Preventive Therapy     : {c['tpt_initiated_count']:,} initiated")
    print(f"  * Secondary Active TB Found  : {c['secondary_tb_detected_count']:,} cases detected")

    print(f"\n[5] DISTRICT DRUG & CONSUMABLE STOCK (drug_stock.csv)")
    print(f"  * Total Stock Log Records    : {s['total_stock_records']:,} (12 months across {s['districts_tracked']} Maharashtra districts)")
    print(f"  * Critical/Low Stock Alerts  : {s['critical_or_low_stock_events']:,} instances flagged for buffer replenishment")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    summary_stats = generate_all_datasets(output_dir="data/synthetic", seed=DEFAULT_SEED)
    print_summary_report(summary_stats)
