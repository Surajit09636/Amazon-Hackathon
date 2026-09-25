from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
import xgboost as xgb

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
try:
    from validation import clean_address, clean_name, extract_digits
except ImportError:
    import re
    def clean_name(t):
        return re.sub(r"[^\w\s]", " ", str(t or "")).lower().strip()
    def clean_address(t):
        return re.sub(r"[^\w\s]", " ", str(t or "")).lower().strip()
    def extract_digits(t):
        return re.findall(r"\b\d+\b", str(t or ""))

BASE_DIR = CURRENT_DIR.parent.parent
DATA_DIR = BASE_DIR / "dataset"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# 1. EVALUATION METRIC (Official Macro F_0.5 with Singletons)


def compute_entity_f05(true_set: set, pred_set: set) -> float:
    """Official F_0.5 calculation for a single entity."""
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0

    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    denom = 0.25 * precision + recall
    return (1.25 * precision * recall) / denom if denom > 0 else 0.0


def evaluate_macro_f05(ground_truth: dict, predictions: dict) -> float:
    """Compute Macro F_0.5 across all S1 entities."""
    total_score = 0.0
    for s1_id, true_matches in ground_truth.items():
        preds = predictions.get(s1_id, set())
        total_score += compute_entity_f05(true_matches, preds)
    return total_score / len(ground_truth) if len(ground_truth) > 0 else 0.0


# -------------------------------------------------------------
# 2. FEATURE EXTRACTION PIPELINE
# -------------------------------------------------------------

FEATURE_NAMES = [
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_ratio",
    "name_exact",
    "name_len_diff",
    "addr_token_set",
    "addr_ratio",
    "addr_len_diff",
    "digit_match_count",
    "digit_jaccard",
    "missing_addr",
    "is_s2",
]


def build_feature_vector(s1_record: tuple, tgt_record: tuple, is_s2: float) -> list[float]:
    """
    Computes rapid pairwise similarity features between S1 and target.
    Record format: (cleaned_name, cleaned_addr, set_of_digits)
    """
    s1_name, s1_addr, s1_digits = s1_record
    tgt_name, tgt_addr, tgt_digits = tgt_record

    # 1. Name features
    name_sort = fuzz.token_sort_ratio(s1_name, tgt_name) / 100.0
    name_set = fuzz.token_set_ratio(s1_name, tgt_name) / 100.0
    name_jw = distance.JaroWinkler.similarity(s1_name, tgt_name)
    name_rat = fuzz.ratio(s1_name, tgt_name) / 100.0
    name_exact = 1.0 if s1_name and s1_name == tgt_name else 0.0
    name_len_diff = float(abs(len(s1_name) - len(tgt_name)))

    # 2. Address features
    missing_addr = 1.0 if not s1_addr or not tgt_addr else 0.0
    if not missing_addr:
        addr_set = fuzz.token_set_ratio(s1_addr, tgt_addr) / 100.0
        addr_rat = fuzz.ratio(s1_addr, tgt_addr) / 100.0
        addr_len_diff = float(abs(len(s1_addr) - len(tgt_addr)))
    else:
        addr_set = 0.0
        addr_rat = 0.0
        addr_len_diff = 0.0

    # 3. Digit/Number overlap features
    intersection = len(s1_digits & tgt_digits)
    union = len(s1_digits | tgt_digits)
    digit_match_count = float(intersection)
    digit_jaccard = float(intersection / union) if union > 0 else 0.0

    return [
        name_sort,
        name_set,
        name_jw,
        name_rat,
        name_exact,
        name_len_diff,
        addr_set,
        addr_rat,
        addr_len_diff,
        digit_match_count,
        digit_jaccard,
        missing_addr,
        is_s2,
    ]


# -------------------------------------------------------------
# 3. TRAINING & THRESHOLD OPTIMIZATION
# -------------------------------------------------------------

def train_and_optimize():
    print("XGBOOST FEATURE EXTRACTION & TRAINING PIPELINE")

    # 1. Load Data
    print("1. Loading validation records and candidate pairs...")
    s1_df = pl.read_csv(VAL_DIR / "val_source1.tsv", separator="\t")
    s2_df = pl.read_csv(VAL_DIR / "val_source2.tsv", separator="\t")
    s3_df = pl.read_csv(VAL_DIR / "val_source3.tsv", separator="\t")
    gt_df = pl.read_csv(VAL_DIR / "val_ground_truth.tsv", separator="\t")

    candidates_path = OUTPUT_DIR / "candidate_pairs.tsv"
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}. Run blocking.py first!")
    cand_df = pl.read_csv(candidates_path, separator="\t")

    # 2. Build fast lookup maps: id -> (cleaned_name, cleaned_addr, set_digits)
    print("2. Preprocessing text fields for fast feature computation...")
    s1_lookup = {}
    for row in s1_df.iter_rows(named=True):
        c_name = clean_name(row["business_name"])
        c_addr = clean_address(row["business_address"])
        digits = set(extract_digits(c_addr))
        s1_lookup[row["entity_id"]] = (c_name, c_addr, digits)

    target_lookup = {}
    for row in s2_df.iter_rows(named=True):
        c_name = clean_name(row["business_name"])
        c_addr = clean_address(row["business_address"])
        digits = set(extract_digits(c_addr))
        target_lookup[row["entity_id"]] = (c_name, c_addr, digits)

    for row in s3_df.iter_rows(named=True):
        c_name = clean_name(row["business_name"])
        c_addr = clean_address(row["business_address"])
        digits = set(extract_digits(c_addr))
        target_lookup[row["entity_id"]] = (c_name, c_addr, digits)

    # 3. Ground Truth Map
    gt_map = {}
    for row in gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip() != "" and str(m_str) != "nan":
            gt_map[s1_id] = set(str(m_str).split(","))
        else:
            gt_map[s1_id] = set()

    # 4. Construct Dataset Matrix X and labels y
    print("3. Building pairwise feature matrix...")
    X_rows = []
    y_labels = []
    pair_metadata = []  # List of (s1_id, cand_id)

    for row in cand_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        cands_str = row["candidate_entity_ids"]
        if not cands_str or str(cands_str).strip() == "" or str(cands_str) == "nan":
            continue

        cands = [c.strip() for c in str(cands_str).split(",") if c.strip()]
        s1_data = s1_lookup.get(s1_id)
        if not s1_data:
            continue

        true_set = gt_map.get(s1_id, set())

        for cand_id in cands:
            tgt_data = target_lookup.get(cand_id)
            if not tgt_data:
                continue

            is_s2 = 1.0 if cand_id.startswith("S2-") else 0.0
            feat_vec = build_feature_vector(s1_data, tgt_data, is_s2)
            label = 1.0 if cand_id in true_set else 0.0

            X_rows.append(feat_vec)
            y_labels.append(label)
            pair_metadata.append((s1_id, cand_id))

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_labels, dtype=np.float32)

    total_pairs = len(y)
    positives = int(y.sum())
    negatives = total_pairs - positives
    print(f"  • Total Candidate Pairs: {total_pairs:,}")
    print(f"  • Positive Pairs (Matches = 1): {positives:,} ({positives/total_pairs*100:.1f}%)")
    print(f"  • Negative Pairs (Look-alikes = 0): {negatives:,} ({negatives/total_pairs*100:.1f}%)")

    # 5. Split for Training and Validation (Group-aware split)
    print("\n4. Training XGBoost Classifier...")
    # Select device: CUDA (RTX 3050) if available, else CPU
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"  • Training Device: {device.upper()} (CUDA Available: {use_cuda})")

    # Train/Val split: 80% train, 20% holdout for threshold tuning
    train_idx, val_idx = train_test_split(
        np.arange(total_pairs),
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Ratio of negative to positive to balance gradient updates
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,  # Standard loss for calibrated probabilities
        tree_method="hist",
        device=device,
        random_state=42,
        eval_metric="logloss",
    )

    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    train_time = time.time() - t0
    print(f"  • Training completed in {train_time:.2f} seconds!")

    # 6. Predict Probabilities Across All Pairs
    print("\n5. Running Inference & Generating Probabilities...")
    all_probs = model.predict_proba(X)[:, 1]

    # Map predictions back to entities
    entity_cand_probs = defaultdict(list)
    for idx, (s1_id, cand_id) in enumerate(pair_metadata):
        prob = float(all_probs[idx])
        entity_cand_probs[s1_id].append((cand_id, prob))

    # 7. Optimize Decision Threshold on Macro F_0.5 Score
    
    print("THRESHOLD OPTIMIZATION FOR MACRO F_0.5")
    

    best_thresh = 0.50
    best_f05 = 0.0

    # Test thresholds from 0.30 to 0.85
    test_thresholds = np.linspace(0.30, 0.85, 12)

    for thresh in test_thresholds:
        # Build predictions dict for this threshold
        predictions = {}
        for s1_id in gt_map.keys():
            cand_list = entity_cand_probs.get(s1_id, [])
            # Only keep candidates above threshold
            filtered_cands = {c for c, p in cand_list if p >= thresh}
            predictions[s1_id] = filtered_cands

        score = evaluate_macro_f05(gt_map, predictions)
        print(f"  Threshold = {thresh:.2f}  -->  Macro F_0.5 = {score:.4f}")

        if score > best_f05:
            best_f05 = score
            best_thresh = float(thresh)

    
    print(f" BEST MACRO F_0.5 SCORE: {best_f05:.4f} (at Threshold = {best_thresh:.2f})")
    

    # 8. Feature Importance
    print("\nTop Predictive Features:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for rank, idx in enumerate(sorted_idx[:8], 1):
        print(f"  {rank}. {FEATURE_NAMES[idx]:<20} : {importances[idx]:.4f}")

    # 9. Save Model & Metadata
    model_file = MODELS_DIR / "xgboost_er_model.json"
    meta_file = MODELS_DIR / "model_config.json"

    model.save_model(str(model_file))
    with open(meta_file, "w") as f:
        json.dump(
            {
                "best_threshold": best_thresh,
                "best_macro_f05": best_f05,
                "feature_names": FEATURE_NAMES,
            },
            f,
            indent=2,
        )

    print(f"\nModel saved to: {model_file}")
    print(f"Config saved to: {meta_file}")


if __name__ == "__main__":
    train_and_optimize()
