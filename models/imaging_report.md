# DEMO MODEL: trained on synthetic cartoon images. Metrics are not evidence of performance.

> [!CAUTION]
> **PIPELINE SMOKE-TEST FIXTURE ONLY**
> This model checkpoint was trained strictly on **synthetic cartoon demo images** (`synthetic_demo_cxr`) to verify software pipeline integration and Grad-CAM visualization mechanics.
> **The metrics below are NOT evidence of diagnostic performance and must never be cited as clinical validation.**
> Real-data training is currently **BLOCKED (needs Kaggle download by user)**.

> [!WARNING]
> **Research prototype, not clinically validated, not for real diagnosis.**
> Diagnostic decisions must always be confirmed by a licensed Medical Officer.

## 1. Overview & Setup
- **Architecture**: Pretrained DenseNet-121 (CheXNet backbone adapted for binary TB screening)
- **Model Provenance**: `synthetic_demo` (trained on `synthetic_demo_cxr` smoke-test fixture)
- **Training Resolution**: `128x128` (Demo mode)
- **Classification Head**: Dropout(0.3) + Linear(1024, 1) + Sigmoid Logit
- **Training Device**: `cpu`
- **Epochs Trained**: `5`
- **Patient-Level Stratified Partitioning (Zero Patient Leakage)**:
  - Training Set: **168** images (70%)
  - Validation Set: **36** images (15%)
  - Held-out Test Set: **36** images (15%)

---

## 2. Test Set Diagnostic Performance (Smoke-Test Fixture Run)

> *Notice: Metrics obtained on synthetic geometric shapes represent pipeline functionality only, NOT diagnostic efficacy.*

| Diagnostic Metric | Smoke-Test Value | Clinical Reality |
| :--- | :--- | :--- |
| **AUC-ROC** | **N/A (Demo Fixture)** | Discriminating geometric circles/ellipses; does not generalize to human anatomy |
| **Sensitivity (Recall)** | **N/A (Demo Fixture)** | Real active pulmonary TB requires clinical validation on human radiographs |
| **Specificity** | **N/A (Demo Fixture)** | Real normal radiograph clearance requires validation on field CXR artifacts |
| **Confusion Matrix** | Evaluated (TN = 18, FP = 0, FN = 0, TP = 18) | Synthetic smoke-test cohort concordance |

*Operating decision threshold: $\tau = 0.50$*

---

## 3. Confusion Matrix (Smoke-Test Test Cohort)

```
                       Actual Normal (0)       Actual TB (1)
Predicted Normal (0)       TN = 18                FN = 0   
Predicted TB (1)           FP = 0                 TP = 18  
```

### Breakdown:
- **True Negatives (TN)**: `18` (Synthetic smoke-test normal images)
- **False Positives (FP)**: `0`
- **False Negatives (FN)**: `0`
- **True Positives (TP)**: `18` (Synthetic smoke-test TB pattern images)

---

## 4. Honest Clinical Discussion & Limitations
1. **No Real Human Validation**: The model evaluated in this report was trained on `synthetic_demo_cxr` cartoon images. Real-data training is BLOCKED until Shenzhen and Montgomery public datasets are downloaded by the user.
2. **Clinical Decision Support Role**: In accordance with Rule 1 of `PROJECT_RULES.md`, this CNN model serves strictly as preliminary decision support. No case can initiate treatment or proceed to follow-up without explicit Medical Officer sign-off.
3. **Interpretability**: Predictions are paired with Grad-CAM saliency heatmaps highlighting apical cavitary or parenchymal regions.
