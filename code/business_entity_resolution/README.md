# Business Entity Resolution Pipeline

This repository contains the complete, production-grade ML pipeline for the **ML Challenge 2026: Business Entity Resolution Challenge**.

## 1. Pipeline Overview
- **Data Normalization (`src/validation.py`):** Unicodedata NFKD normalization, ftfy text repair, corporate legal suffix stripping (e.g. `Pvt Ltd`, `SARL`, `LLC`, `Corp`), street abbreviation standardizations, and digit isolation.
- **Candidate Generation (`src/inference.py`):** Multi-key inverted index using:
  1. Name 5-character and 4-character compact prefixes
  2. Significant name word tokens (excluding corporate legal stopwords)
  3. Postal codes (5-digit US and 6-digit India)
  4. Building numbers and street roots
  - Ranked using SIMD AVX2 `RapidFuzz` token similarity before scoring.
  - Two-pass source separation (Source 2 then Source 3) strictly capping memory consumption **< 1.1 GB RAM** with zero pagefile swapping.
- **Matching Model (`src/train_model.py`):** Gradient-boosted decision trees (`XGBoost`) trained on 13 pairwise string, token-sort, token-set, Jaro-Winkler, address ratio, digit overlap, and source indicator features.
- **Decision Threshold:** Calibrated at **0.65** on holdout validation data, achieving **0.9738 Macro $F_{0.5}$** on validation pairs.

## 2. Directory Structure
```
code/business_entity_resolution/
├── src/
│   ├── validation.py        # Normalization, cleaning, and validation split generator
│   ├── blocking.py          # Candidate generation / blocking engine
│   ├── train_model.py       # XGBoost ER training, threshold tuning, and feature extractor
│   ├── inference.py         # End-to-end ultra-lean test inference script (< 1.1 GB RAM)
│   └── eda.py               # Exploratory data analysis and singleton distribution audit
├── models/
│   ├── xgboost_er_model.json # Serialized XGBoost model
│   └── model_config.json    # Optimal threshold and feature configuration
├── requirements.txt         # Pinned python environment dependencies
└── README.md                # Reproduction and execution guide
```

## 3. Environment Setup
Install dependencies in a Python 3.12 virtual environment:
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

## 4. How to Reproduce End-to-End

### Step 1: Run Training & Threshold Tuning (Optional)
To retrain the XGBoost entity resolution model on the training ground truth:
```bash
python src/train_model.py
```
This saves the trained model to `models/xgboost_er_model.json` and evaluates Macro $F_{0.5}$.

### Step 2: Run Test Inference
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` across all 1,732,544 test entities:
```bash
python src/inference.py
```
*Note: Memory is strictly capped under 1.1 GB RAM throughout execution.*

### Step 3: Validate Submission
Validate the generated output files with the competition checker:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```
