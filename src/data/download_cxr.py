"""Public Chest X-ray Dataset Downloader and Benchmark Curator.

Handles:
1. Automated download of Montgomery & Shenzhen TB datasets via Kaggle API if credentials exist.
2. Formatted manual download instructions if Kaggle credentials are not configured.
3. Automated curation of synthetic_demo_cxr samples (pipeline smoke-test fixture only).
4. Raw data processing (--process) building metadata.csv with dataset_source in {shenzhen, montgomery}
   and verified filename-label convention (_0 normal / _1 TB).

DISCLAIMER:
Research prototype, not clinically validated, not for real diagnosis.
"""

import argparse
import csv
import math
import os
import random
import shutil
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

DATA_DIR = os.path.join("data", "cxr")
RAW_DIR = os.path.join(DATA_DIR, "raw")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
DEMO_DIR = os.path.join(DATA_DIR, "synthetic_demo_cxr")
LEGACY_BENCHMARK_DIR = os.path.join(DATA_DIR, "benchmark")


def check_kaggle_credentials() -> bool:
    """Check whether Kaggle API credentials are present."""
    home = os.path.expanduser("~")
    kaggle_json = os.path.join(home, ".kaggle", "kaggle.json")
    if os.path.exists(kaggle_json):
        return True
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return False


def get_manual_download_instructions() -> str:
    """Return explicit manual download instructions for Kaggle CXR datasets."""
    return f"""
================================================================================
           PUBLIC CHEST X-RAY TB DATA - MANUAL DOWNLOAD INSTRUCTIONS
================================================================================
Kaggle API credentials were not detected in ~/.kaggle/kaggle.json or environment.

Please follow these steps to download the official public TB CXR datasets:

1. SHENZHEN HOSPITAL TB DATASET:
   - URL: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-shenzhen
   - Alternative: https://www.kaggle.com/datasets/kmader/pulmonary-chest-xray-abnormalities
   - Download the archive (approx 1.2 GB)
   - Extract images into:
     '{os.path.abspath(os.path.join(RAW_DIR, "shenzhen"))}'
   - Verified naming convention:
     Filenames ending with '_0.png' are Normal (e.g., CHNCXR_0001_0.png)
     Filenames ending with '_1.png' are Tuberculosis (e.g., CHNCXR_0350_1.png)

2. MONTGOMERY COUNTY TB DATASET:
   - URL: https://www.kaggle.com/datasets/raddar/tuberculosis-chest-xrays-montgomery
   - Alternative: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Montgomery-County-CXR-Set/MontgomerySet.zip
   - Download the archive (approx 500 MB)
   - Extract images into:
     '{os.path.abspath(os.path.join(RAW_DIR, "montgomery"))}'
   - Verified naming convention:
     Filenames ending with '_0.png' are Normal (e.g., MCUCXR_0001_0.png)
     Filenames ending with '_1.png' are Tuberculosis (e.g., MCUCXR_0050_1.png)

3. Once downloaded, process and index into metadata.csv:
   python src/data/download_cxr.py --process
================================================================================
"""


def try_kaggle_download() -> bool:
    """Attempt download using Kaggle CLI or Python API."""
    if not check_kaggle_credentials():
        return False

    try:
        import kaggle  # type: ignore
        os.makedirs(os.path.join(RAW_DIR, "shenzhen"), exist_ok=True)
        os.makedirs(os.path.join(RAW_DIR, "montgomery"), exist_ok=True)
        print("[Kaggle API] Authenticated. Downloading Shenzhen TB dataset...")
        kaggle.api.dataset_download_files(
            "raddar/tuberculosis-chest-xrays-shenzhen",
            path=os.path.join(RAW_DIR, "shenzhen"),
            unzip=True
        )
        print("[Kaggle API] Downloading Montgomery TB dataset...")
        kaggle.api.dataset_download_files(
            "raddar/tuberculosis-chest-xrays-montgomery",
            path=os.path.join(RAW_DIR, "montgomery"),
            unzip=True
        )
        return True
    except Exception as e:
        print(f"[Kaggle API Error] Download failed: {e}")
        return False


def create_synthetic_demo_cxr_dataset(num_samples: int = 240, img_size: int = 128) -> str:
    """Create synthetic_demo_cxr images (smoke-test fixture only).

    WARNING:
    This dataset is strictly a software integration smoke-test fixture.
    It does not contain real human radiographs and cannot be used for clinical validation.
    """
    from PIL import Image, ImageDraw, ImageFilter

    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(LEGACY_BENCHMARK_DIR, exist_ok=True)

    records = []
    random.seed(42)

    half = num_samples // 2

    for i in range(num_samples):
        is_tb = 1 if i >= half else 0
        img_id = f"CXR_{'TB' if is_tb else 'NORMAL'}_{i:04d}.png"
        img_path = os.path.join(DEMO_DIR, img_id)
        legacy_path = os.path.join(LEGACY_BENCHMARK_DIR, img_id)

        # Generate base synthetic radiograph
        img = Image.new("L", (img_size, img_size), color=25)
        draw = ImageDraw.Draw(img)

        # Thoracic cavity outer rib boundary (oval)
        cx, cy = img_size // 2, img_size // 2
        draw.ellipse([cx - 52, cy - 50, cx + 52, cy + 50], fill=45, outline=70)

        # Left and Right Lung Fields
        draw.ellipse([cx - 46, cy - 40, cx - 10, cy + 38], fill=15, outline=30)
        draw.ellipse([cx + 10, cy - 40, cx + 46, cy + 38], fill=15, outline=30)

        # Mediastinum and Cardiac Silhouette
        draw.ellipse([cx - 14, cy - 45, cx + 14, cy + 45], fill=140, outline=160)
        draw.ellipse([cx - 12, cy - 5, cx + 24, cy + 32], fill=170, outline=190)

        # Clavicles
        draw.line([cx - 50, cy - 35, cx - 12, cy - 32], fill=160, width=3)
        draw.line([cx + 12, cy - 32, cx + 50, cy - 35], fill=160, width=3)

        # Add TB specific pathologic patterns if label == 1
        if is_tb:
            draw.ellipse([cx - 38, cy - 34, cx - 22, cy - 18], fill=210, outline=235)
            draw.ellipse([cx - 34, cy - 30, cx - 26, cy - 22], fill=30)

            for _ in range(6):
                px = random.randint(cx - 36, cx - 16)
                py = random.randint(cy - 25, cy + 5)
                rad = random.randint(3, 7)
                draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=random.randint(180, 220))

            if random.random() < 0.6:
                for _ in range(4):
                    px = random.randint(cx + 18, cx + 36)
                    py = random.randint(cy - 28, cy - 10)
                    rad = random.randint(3, 6)
                    draw.ellipse([px - rad, py - rad, px + rad, py + rad], fill=random.randint(170, 210))

        # Soft tissue scatter smoothing
        img = img.filter(ImageFilter.GaussianBlur(radius=1.2))

        # Detector noise
        pixels = img.load()
        for x in range(img_size):
            for y in range(img_size):
                noise = random.randint(-4, 4)
                val = max(0, min(255, pixels[x, y] + noise))
                pixels[x, y] = val

        # Save to both synthetic_demo_cxr and legacy benchmark paths
        rgb_img = img.convert("RGB")
        rgb_img.save(img_path)
        rgb_img.save(legacy_path)

        records.append({
            "image_id": img_id,
            "image_path": img_path.replace("\\", "/"),
            "label": is_tb,
            "dataset_source": "synthetic_demo_cxr",
            "findings": "Synthetic cartoon demo radiograph: apical consolidation" if is_tb else "Synthetic cartoon demo radiograph: clear fields",
            "patient_id": f"SYNTH_{i:04d}",
        })

    # Write metadata index
    with open(METADATA_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "image_path", "label", "dataset_source", "findings", "patient_id"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"[synthetic_demo_cxr] Generated {num_samples} smoke-test images in {DEMO_DIR}")
    print(f"[Metadata] Indexed in {METADATA_FILE} with dataset_source=synthetic_demo_cxr")
    return METADATA_FILE


# Alias for legacy compatibility
create_benchmark_cxr_dataset = create_synthetic_demo_cxr_dataset


def parse_filename_label(filename: str) -> Optional[Tuple[int, str]]:
    """Parse filename using the official _0 (normal) / _1 (TB) convention.

    Returns:
        (label, patient_id) if valid filename convention, else None.
    """
    base, ext = os.path.splitext(filename)
    if ext.lower() not in [".png", ".jpg", ".jpeg"]:
        return None

    if base.endswith("_0"):
        label = 0
        patient_id = base[:-2]
        return label, patient_id
    elif base.endswith("_1"):
        label = 1
        patient_id = base[:-2]
        return label, patient_id

    return None


def scan_and_index_raw_data() -> Optional[str]:
    """Scan data/cxr/raw/ for downloaded Shenzhen & Montgomery images and build metadata.csv.

    Verifies the official filename convention (_0 normal / _1 TB) and assigns
    dataset_source in {"shenzhen", "montgomery"}.
    """
    records = []
    skipped_files = []

    # Check Shenzhen
    sz_dir = os.path.join(RAW_DIR, "shenzhen")
    if os.path.exists(sz_dir):
        for root, _, files in os.walk(sz_dir):
            for file in files:
                parsed = parse_filename_label(file)
                if parsed:
                    label, patient_id = parsed
                    records.append({
                        "image_id": file,
                        "image_path": os.path.join(root, file).replace("\\", "/"),
                        "label": label,
                        "dataset_source": "shenzhen",
                        "findings": "Shenzhen public CXR",
                        "patient_id": patient_id,
                    })
                elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                    skipped_files.append((file, "shenzhen"))

    # Check Montgomery
    mc_dir = os.path.join(RAW_DIR, "montgomery")
    if os.path.exists(mc_dir):
        for root, _, files in os.walk(mc_dir):
            for file in files:
                parsed = parse_filename_label(file)
                if parsed:
                    label, patient_id = parsed
                    records.append({
                        "image_id": file,
                        "image_path": os.path.join(root, file).replace("\\", "/"),
                        "label": label,
                        "dataset_source": "montgomery",
                        "findings": "Montgomery public CXR",
                        "patient_id": patient_id,
                    })
                elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                    skipped_files.append((file, "montgomery"))

    if skipped_files:
        print(f"[Warning] Skipped {len(skipped_files)} files not matching _0 / _1 convention.")

    if records:
        with open(METADATA_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["image_id", "image_path", "label", "dataset_source", "findings", "patient_id"]
            )
            writer.writeheader()
            writer.writerows(records)

        sz_count = sum(1 for r in records if r["dataset_source"] == "shenzhen")
        mc_count = sum(1 for r in records if r["dataset_source"] == "montgomery")
        print(f"[Scan] Successfully indexed {len(records)} real CXR images into {METADATA_FILE}")
        print(f"       Shenzhen: {sz_count} images | Montgomery: {mc_count} images")
        return METADATA_FILE

    return None


def is_real_data_present() -> bool:
    """Check if real Shenzhen or Montgomery CXR datasets are present in metadata or raw directory."""
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                sources = {r.get("dataset_source") for r in reader}
                if any(s in {"shenzhen", "montgomery"} for s in sources):
                    return True
        except Exception:
            pass

    # Check raw directories
    for dataset in ["shenzhen", "montgomery"]:
        d = os.path.join(RAW_DIR, dataset)
        if os.path.exists(d):
            for _, _, files in os.walk(d):
                if any(parse_filename_label(f) for f in files):
                    return True
    return False


def setup_cxr_data() -> str:
    """Primary entry point: check raw datasets, attempt Kaggle, or create synthetic demo dataset."""
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1. Check if raw real datasets are already present
    indexed = scan_and_index_raw_data()
    if indexed:
        return indexed

    # 2. Try Kaggle API
    if try_kaggle_download():
        indexed = scan_and_index_raw_data()
        if indexed:
            return indexed

    # 3. Print manual download instructions
    print(get_manual_download_instructions())

    # 4. Generate synthetic_demo_cxr smoke-test dataset so pipeline tests run immediately
    print("[Downloader] Real data absent: generating synthetic_demo_cxr smoke-test fixture...")
    print("[Downloader] Note: Real-data training is BLOCKED until real datasets are downloaded.")
    return create_synthetic_demo_cxr_dataset(num_samples=240, img_size=128)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chest X-Ray Data Downloader and Curator")
    parser.add_argument("--process", action="store_true", help="Process raw downloaded datasets and build metadata.csv")
    parser.add_argument("--demo", action="store_true", help="Force generate synthetic_demo_cxr smoke-test dataset")
    args = parser.parse_args()

    if args.process:
        indexed = scan_and_index_raw_data()
        if indexed:
            print(f"[Done] Real data metadata generated: {indexed}")
        else:
            print("[BLOCKED] Real data absent in data/cxr/raw/shenzhen/ and data/cxr/raw/montgomery/.")
            print("Needs Kaggle download by user. Run without --process to see download instructions.")
    elif args.demo:
        create_synthetic_demo_cxr_dataset()
    else:
        setup_cxr_data()
