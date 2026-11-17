"""CNN Training and Evaluation Script for Tuberculosis Chest Radiographs.

Fine-tunes a pretrained DenseNet121 architecture (CheXNet backbone) on CXR data.
Features:
- Refuses to write a "validated" report on synthetic_demo_cxr data; marks it clearly as a smoke-test fixture.
- If real data is absent, marks real-data training BLOCKED (needs Kaggle download by user).
- Patient-level stratified split with zero patient leakage across train / val / test.
- Comprehensive test set evaluation: AUC-ROC with bootstrap 95% CI, Sensitivity, Specificity, F1, and Confusion Matrix.
- 224x224 training on real data with CPU-friendly defaults.
- Markdown clinical report saved to models/imaging_report.md.
- Checkpoint persistence saved to models/imaging_densenet121.pt.

DISCLAIMER:
Research prototype, not clinically validated, not for real diagnosis.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

from src.data.download_cxr import METADATA_FILE, is_real_data_present, setup_cxr_data

PROVENANCE_FILE = os.path.join("models", "model_provenance.json")


class RealDataBlockedError(RuntimeError):
    """Raised when real-data training is attempted but real public datasets are absent."""
    pass


class ChestXRayDataset(Dataset):
    """PyTorch Dataset loading chest radiographs and binary TB labels."""

    def __init__(self, records: List[Dict[str, Any]], transform=None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.records[idx]
        img_path = item["image_path"]
        label = float(item["label"])

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback for missing/corrupted file
            image = Image.new("RGB", (128, 128), color=50)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = T.ToTensor()(image)

        label_tensor = torch.tensor(label, dtype=torch.float32)
        return image_tensor, label_tensor, img_path


def get_transforms(img_size: int = 224) -> Tuple[Any, Any]:
    """Define data augmentation and normalization pipelines."""
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def build_densenet121_classifier(pretrained: bool = True) -> nn.Module:
    """Build DenseNet121 architecture with a binary classification head."""
    try:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
    except Exception:
        model = models.densenet121(weights=None)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 1)
    )
    return model


def patient_stratified_split(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform patient-level stratified split ensuring zero patient leakage across folds."""
    # Group records by patient_id
    patient_to_records: Dict[str, List[Dict[str, Any]]] = {}
    patient_to_label: Dict[str, int] = {}

    for r in records:
        pid = r.get("patient_id") or r.get("image_id")
        patient_to_records.setdefault(pid, []).append(r)
        patient_to_label[pid] = int(r["label"])

    pos_patients = [pid for pid, lbl in patient_to_label.items() if lbl == 1]
    neg_patients = [pid for pid, lbl in patient_to_label.items() if lbl == 0]

    rnd = random.Random(seed)
    rnd.shuffle(pos_patients)
    rnd.shuffle(neg_patients)

    def split_pids(pids: List[str]):
        n = len(pids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return pids[:n_train], pids[n_train:n_train + n_val], pids[n_train + n_val:]

    tr_pos, val_pos, te_pos = split_pids(pos_patients)
    tr_neg, val_neg, te_neg = split_pids(neg_patients)

    train_pids = set(tr_pos + tr_neg)
    val_pids = set(val_pos + val_neg)
    test_pids = set(te_pos + te_neg)

    train_records = [r for pid in train_pids for r in patient_to_records[pid]]
    val_records = [r for pid in val_pids for r in patient_to_records[pid]]
    test_records = [r for pid in test_pids for r in patient_to_records[pid]]

    rnd.shuffle(train_records)
    rnd.shuffle(val_records)
    rnd.shuffle(test_records)

    return train_records, val_records, test_records


# Backward compatibility alias
stratified_split = patient_stratified_split


def compute_roc_auc(y_true: List[int], y_scores: List[float]) -> float:
    """Compute exact Area Under the ROC Curve using Mann-Whitney U rank statistic."""
    pos_scores = [s for y, s in zip(y_true, y_scores) if y == 1]
    neg_scores = [s for y, s in zip(y_true, y_scores) if y == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    wins = 0.0
    for ps in pos_scores:
        for ns in neg_scores:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5

    return wins / (n_pos * n_neg)


def compute_auc_bootstrap_ci(
    y_true: List[int],
    y_scores: List[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Tuple[float, float, float]:
    """Compute bootstrap 95% confidence interval for AUC-ROC."""
    rnd = random.Random(seed)
    n = len(y_true)
    if n < 4:
        auc = compute_roc_auc(y_true, y_scores)
        return auc, auc, auc

    bootstrapped_aucs: List[float] = []
    indices = list(range(n))

    for _ in range(n_bootstraps):
        resampled_idx = [rnd.choice(indices) for _ in range(n)]
        b_true = [y_true[i] for i in resampled_idx]
        b_scores = [y_scores[i] for i in resampled_idx]
        # Only compute if both positive and negative classes are sampled
        if len(set(b_true)) < 2:
            continue
        bootstrapped_aucs.append(compute_roc_auc(b_true, b_scores))

    if not bootstrapped_aucs:
        point_auc = compute_roc_auc(y_true, y_scores)
        return point_auc, point_auc, point_auc

    bootstrapped_aucs.sort()
    lower_idx = int((alpha / 2.0) * len(bootstrapped_aucs))
    upper_idx = int((1.0 - alpha / 2.0) * len(bootstrapped_aucs))
    upper_idx = min(upper_idx, len(bootstrapped_aucs) - 1)

    point_auc = compute_roc_auc(y_true, y_scores)
    ci_lower = bootstrapped_aucs[lower_idx]
    ci_upper = bootstrapped_aucs[upper_idx]
    return point_auc, ci_lower, ci_upper


def compute_classification_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float = 0.5,
    n_bootstraps: int = 500
) -> Dict[str, Any]:
    """Compute complete classification metrics: AUC with bootstrap 95% CI, Sensitivity, Specificity, F1, and Confusion Matrix."""
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    auc, auc_ci_lower, auc_ci_upper = compute_auc_bootstrap_ci(y_true, y_scores, n_bootstraps=n_bootstraps)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc,
        "auc_ci_lower": auc_ci_lower,
        "auc_ci_upper": auc_ci_upper,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": threshold,
    }


def train_imaging_model(
    metadata_path: str = METADATA_FILE,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    img_size: Optional[int] = None,
    save_model_path: str = "models/imaging_densenet121.pt",
    report_path: str = "models/imaging_report.md",
    seed: int = 42,
    require_real: bool = False
) -> Dict[str, Any]:
    """Execute fine-tuning pipeline for DenseNet121 TB classifier.

    - If require_real is True and real data is absent, marks training BLOCKED.
    - If dataset_source == 'synthetic_demo_cxr', refuses to write a validated report
      and writes the DEMO MODEL smoke-test notice instead.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Training device: {device}")

    # Ensure dataset is indexed
    if not os.path.exists(metadata_path):
        metadata_path = setup_cxr_data()

    records = []
    with open(metadata_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)

    if not records:
        raise RuntimeError("No CXR records found in metadata index.")

    sources = set(r.get("dataset_source", "") for r in records)
    is_real = any(s in {"shenzhen", "montgomery"} for s in sources)
    is_synthetic = all(s in {"synthetic_demo_cxr", "benchmark_synthetic_cxr"} for s in sources)

    if require_real and not is_real:
        raise RealDataBlockedError(
            "[BLOCKED] Real dataset absent in data/cxr/raw/shenzhen/ or data/cxr/raw/montgomery/. "
            "Needs Kaggle download by user. Synthetic demo data cannot be substituted for real-data training."
        )

    # Resolution: 224x224 on real data, 128x128 on synthetic demo
    if img_size is None:
        img_size = 224 if is_real else 128

    provenance = "real_public_data" if is_real else "synthetic_demo"
    print(f"[Provenance] Dataset source(s): {sources} | Provenance: {provenance}")
    print(f"[Image Resolution] Training at {img_size}x{img_size}")

    train_records, val_records, test_records = patient_stratified_split(records, seed=seed)
    print(f"  * Train set : {len(train_records)} images")
    print(f"  * Val set   : {len(val_records)} images")
    print(f"  * Test set  : {len(test_records)} images")

    train_tf, eval_tf = get_transforms(img_size=img_size)
    train_ds = ChestXRayDataset(train_records, transform=train_tf)
    val_ds = ChestXRayDataset(val_records, transform=eval_tf)
    test_ds = ChestXRayDataset(test_records, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    pos_count = sum(1 for r in train_records if int(r["label"]) == 1)
    neg_count = sum(1 for r in train_records if int(r["label"]) == 0)
    pos_weight_val = neg_count / max(1, pos_count)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)

    model = build_densenet121_classifier(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    print(f"\n[Training] Starting DenseNet121 fine-tuning ({provenance})...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= max(1, len(train_records))

        # Validation
        model.eval()
        val_loss = 0.0
        val_true = []
        val_scores = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits).cpu().tolist()
                val_scores.extend(probs if isinstance(probs, list) else [probs])
                val_true.extend(labels.cpu().tolist())

        val_loss /= max(1, len(val_records))
        val_auc = compute_roc_auc(val_true, val_scores)

        print(f"  Epoch [{epoch}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model_path)

    print(f"[Training Complete] Elapsed time: {time.time() - t_start:.1f}s")
    print(f"[Model Saved] Best weights checkpoint saved to: {save_model_path}")

    # Evaluate on held-out test set
    if os.path.exists(save_model_path):
        model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()

    test_true = []
    test_scores = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().tolist()
            test_scores.extend(probs if isinstance(probs, list) else [probs])
            test_true.extend(labels.cpu().tolist())

    metrics = compute_classification_metrics(test_true, test_scores, threshold=0.5)

    # Save model provenance metadata
    save_model_provenance(provenance, save_model_path)

    # Write evaluation report
    save_imaging_report(
        metrics=metrics,
        train_count=len(train_records),
        val_count=len(val_records),
        test_count=len(test_records),
        epochs=epochs,
        device=str(device),
        img_size=img_size,
        provenance=provenance,
        report_path=report_path
    )

    return metrics


def save_model_provenance(provenance: str, model_path: str):
    """Save provenance metadata JSON alongside model checkpoint."""
    prov_data = {
        "model_provenance": provenance,
        "model_path": model_path,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "Research prototype, not clinically validated, not for real diagnosis."
    }
    os.makedirs(os.path.dirname(PROVENANCE_FILE), exist_ok=True)
    with open(PROVENANCE_FILE, mode="w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)


def save_imaging_report(
    metrics: Dict[str, Any],
    train_count: int,
    val_count: int,
    test_count: int,
    epochs: int,
    device: str,
    img_size: int,
    provenance: str,
    report_path: str = "models/imaging_report.md"
):
    """Write markdown evaluation report.

    REFUSES to write a 'validated' report when dataset is synthetic_demo_cxr.
    Instead heads the report with the required DEMO MODEL notice.
    """
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    if provenance == "synthetic_demo":
        header = "# DEMO MODEL: trained on synthetic cartoon images. Metrics are not evidence of performance."
        status_banner = """
> [!CAUTION]
> **PIPELINE SMOKE-TEST FIXTURE ONLY**
> This model was trained strictly on **synthetic cartoon demo images** (`synthetic_demo_cxr`) for software pipeline integration.
> **The metrics below are NOT evidence of diagnostic performance and must never be cited as clinical validation.**
> Real-data training is currently **BLOCKED (needs Kaggle download by user)**.
"""
    else:
        header = "# Chest X-ray Imaging Agent: Real-Data Model Evaluation Report"
        status_banner = """
> [!WARNING]
> **Research prototype, not clinically validated, not for real diagnosis.**
> Trained on public chest radiograph benchmarks (Shenzhen / Montgomery).
> Diagnostic decisions must always be confirmed by a licensed Medical Officer via the Doctor Review Gate.
"""

    ci_str = f" [{metrics['auc_ci_lower']:.4f} - {metrics['auc_ci_upper']:.4f}]" if 'auc_ci_lower' in metrics else ""

    report_content = f"""{header}
{status_banner}
## 1. Overview & Setup
- **Architecture**: Pretrained DenseNet-121 (CheXNet backbone adapted for binary TB screening)
- **Model Provenance**: `{provenance}`
- **Training Resolution**: `{img_size}x{img_size}`
- **Classification Head**: Dropout(0.3) + Linear(1024, 1) + Sigmoid Logit
- **Training Device**: `{device}`
- **Epochs Trained**: `{epochs}`
- **Evaluation Timestamp**: `{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}`
- **Patient-Level Stratified Partitioning (Zero Patient Leakage)**:
  - Training Set: **{train_count}** images (70%)
  - Validation Set: **{val_count}** images (15%)
  - Held-out Test Set: **{test_count}** images (15%)

---

## 2. Test Set Diagnostic Performance

| Diagnostic Metric | Value | 95% Bootstrap CI | Clinical Interpretation |
| :--- | :--- | :--- | :--- |
| **AUC-ROC** | **{metrics['auc']:.4f}** | `{ci_str.strip() or 'N/A'}` | Discriminative ranking performance across test set |
| **Sensitivity (Recall)** | **{metrics['sensitivity']:.4f}** ({metrics['sensitivity']*100:.1f}%) | - | Active pulmonary TB detection rate at operating threshold |
| **Specificity** | **{metrics['specificity']:.4f}** ({metrics['specificity']*100:.1f}%) | - | Normal radiograph clearance rate without false alarms |
| **Precision (PPV)** | **{metrics['precision']:.4f}** ({metrics['precision']*100:.1f}%) | - | Proportion of positive predictions that are true TB |
| **F1 Score** | **{metrics['f1']:.4f}** | - | Harmonic mean of sensitivity and precision |
| **Overall Accuracy** | **{metrics['accuracy']:.4f}** ({metrics['accuracy']*100:.1f}%) | - | Overall concordance across test cohort |

*Operating decision threshold: $\\tau = {metrics['threshold']:.2f}$*

---

## 3. Confusion Matrix (Held-out Test Cohort)

```
                       Actual Normal (0)       Actual TB (1)
Predicted Normal (0)       TN = {metrics['tn']:<4}              FN = {metrics['fn']:<4}
Predicted TB (1)           FP = {metrics['fp']:<4}              TP = {metrics['tp']:<4}
```

### Breakdown:
- **True Negatives (TN)**: `{metrics['tn']}` (Correctly identified as normal)
- **False Positives (FP)**: `{metrics['fp']}` (Flagged for confirmatory CBNAAT / GeneXpert testing)
- **False Negatives (FN)**: `{metrics['fn']}` (Missed; protected by symptom intake and doctor review gate)
- **True Positives (TP)**: `{metrics['tp']}` (Accurately flagged with parenchymal radiopacity)

---

## 4. Honest Clinical Discussion & Limitations
1. **Provenance & Dataset Reality**: {"Trained on synthetic cartoon demo images. Real-data training is BLOCKED until real Shenzhen/Montgomery CXRs are downloaded by user." if provenance == "synthetic_demo" else "Trained on public reference datasets (Shenzhen & Montgomery). Domain shift exists between hospital datasets and frontline rural mobile screening vans."}
2. **Clinical Decision Support Role**: In accordance with Rule 1 of `PROJECT_RULES.md`, this CNN model serves strictly as preliminary decision support. No case can initiate treatment or proceed to follow-up without explicit Medical Officer sign-off.
3. **Interpretability**: Predictions are paired with Grad-CAM saliency heatmaps highlighting apical cavitary or parenchymal regions.
"""

    with open(report_path, mode="w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[Report Saved] Evaluation report written to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet121 TB CXR Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=None, help="Image resolution (default 224 for real, 128 for demo)")
    parser.add_argument("--real-data", action="store_true", help="Require real dataset (fails if real data absent)")
    args = parser.parse_args()

    train_imaging_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        require_real=args.real_data
    )
