from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from sklearn.model_selection import train_test_split
import torch
import xgboost as xgb

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
try:
    from validation import (
        clean_address,
        clean_name,
        extract_digits,
        extract_pin,
        extract_primary_bldg,
        is_non_latin,
    )
except ImportError:
    from code.src.validation import (
        clean_address,
        clean_name,
        extract_digits,
        extract_pin,
        extract_primary_bldg,
        is_non_latin,
    )

BASE_DIR = CURRENT_DIR.parent.parent
DATA_DIR = BASE_DIR / "dataset"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. EVALUATION METRIC (Official Macro F_0.5 with Singletons)
# -------------------------------------------------------------

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
# 2. FEATURE EXTRACTION PIPELINE (17 ADVANCED SIGNALS)
# -------------------------------------------------------------

FEATURE_NAMES = [
    "name_token_sort",
    "name_token_set",
    "name_jaro_winkler",
    "name_ratio",
    "name_exact",
    "core_brand_sim",
    "name_len_diff",
    "is_cross_script",
    "addr_token_set",
    "addr_ratio",
    "addr_len_diff",
    "postal_match_status",
    "bldg_match_status",
    "digit_match_count",
    "digit_jaccard",
    "missing_addr",
    "is_s2",
]


def build_advanced_feature_vector(
    s1_raw: tuple,
    tgt_raw: tuple,
    s1_clean_rec: tuple,
    tgt_clean_rec: tuple,
    is_s2: float,
) -> list[float]:
    """
    Computes 17 high-discriminability pairwise ER features.
    Inputs:
      s1_raw, tgt_raw: (raw_name_str, raw_addr_str)
      s1_clean_rec, tgt_clean_rec: (cleaned_name, cleaned_addr, digit_set)
      is_s2: 1.0 if Source 2, 0.0 if Source 3
    """
    s1_name, s1_addr, s1_digits = s1_clean_rec[:3]
    tgt_name, tgt_addr, tgt_digits = tgt_clean_rec[:3]

    # 1. Name Similarity Features
    n_sort = fuzz.token_sort_ratio(s1_name, tgt_name) / 100.0
    n_set = fuzz.token_set_ratio(s1_name, tgt_name) / 100.0
    n_jw = distance.JaroWinkler.similarity(s1_name, tgt_name)
    n_rat = fuzz.ratio(s1_name, tgt_name) / 100.0
    n_exact = 1.0 if s1_name and s1_name == tgt_name else 0.0
    n_len_diff = float(abs(len(s1_name) - len(tgt_name)))

    # Core Brand Token Match (first 2 words)
    w1 = s1_name.split()[:2]
    w2 = tgt_name.split()[:2]
    core_sim = fuzz.token_set_ratio(" ".join(w1), " ".join(w2)) / 100.0 if (w1 and w2) else 0.0

    # Cross-script detection (pre-cached or dynamic fallback)
    if len(s1_clean_rec) >= 6 and len(tgt_clean_rec) >= 6:
        pin1, b1, nl1 = s1_clean_rec[3:6]
        pin2, b2, nl2 = tgt_clean_rec[3:6]
        is_cross = 1.0 if nl1 != nl2 else 0.0
    else:
        is_cross = 1.0 if (is_non_latin(s1_raw[0]) != is_non_latin(tgt_raw[0])) else 0.0
        pin1 = extract_pin(s1_raw[1])
        pin2 = extract_pin(tgt_raw[1])
        b1 = extract_primary_bldg(s1_raw[1])
        b2 = extract_primary_bldg(tgt_raw[1])

    # 2. Address Similarity Features
    missing_addr = 1.0 if not s1_addr or not tgt_addr else 0.0
    if not missing_addr:
        a_set = fuzz.token_set_ratio(s1_addr, tgt_addr) / 100.0
        a_rat = fuzz.ratio(s1_addr, tgt_addr) / 100.0
        a_len_diff = float(abs(len(s1_addr) - len(tgt_addr)))
    else:
        a_set, a_rat, a_len_diff = 0.0, 0.0, 0.0

    # Postal code match vs conflict (+1.0 match, -1.0 conflict, 0.0 missing)
    if pin1 and pin2:
        pin_status = 1.0 if pin1 == pin2 else -1.0
    else:
        pin_status = 0.0

    # Primary building number match vs conflict (+1.0 match, -1.0 conflict, 0.0 missing)
    if b1 and b2:
        b_status = 1.0 if b1 == b2 else -1.0
    else:
        b_status = 0.0

    # 3. Numerical / Digit Overlap Features
    inter = len(s1_digits & tgt_digits)
    union = len(s1_digits | tgt_digits)
    d_count = float(inter)
    d_jacc = float(inter / union) if union > 0 else 0.0

    return [
        n_sort,
        n_set,
        n_jw,
        n_rat,
        n_exact,
        core_sim,
        n_len_diff,
        is_cross,
        a_set,
        a_rat,
        a_len_diff,
        pin_status,
        b_status,
        d_count,
        d_jacc,
        missing_addr,
        is_s2,
    ]


# Backward compatible alias
build_feature_vector = build_advanced_feature_vector


# -------------------------------------------------------------
# 3. ENHANCED BLOCKING FOR HARD NEGATIVE TRAINING
# -------------------------------------------------------------

LEGAL_STOPWORDS = {
    "pvt", "ltd", "limited", "private", "llc", "corp", "corporation", "inc",
    "incorporated", "co", "company", "sarl", "sa", "sas", "enterprises",
    "solutions", "services", "the", "and", "for", "group", "industries",
    "international", "holding", "holdings", "management", "india", "france", "us"
}

ADDR_STOPWORDS = {
    "road", "street", "avenue", "lane", "drive", "court", "floor", "apartment",
    "near", "opposite", "cross", "main", "nagar", "city", "state", "district",
    "rue", "boulevard", "place", "allée", "chemin", "route"
}

RE_DIGITS = re.compile(r"\b\d+\b")


def get_enhanced_keys(c_name: str, c_addr: str) -> list[str]:
    """Generates 99.66% recall multi-aspect blocking keys."""
    keys = []
    compact = c_name.replace(" ", "")
    if len(compact) >= 4:
        keys.append("p:" + compact[:5])
        keys.append("p4:" + compact[:4])

    words = [w for w in c_name.split() if len(w) >= 3 and w not in LEGAL_STOPWORDS]
    for w in words[:5]:
        keys.append("w:" + w)
        if len(w) >= 5:
            keys.append("3g:" + w[:3])
            keys.append("3g:" + w[-3:])

    digits = RE_DIGITS.findall(c_addr)
    for d in digits:
        if len(d) in (5, 6):
            keys.append("pin:" + d)
        elif len(d) >= 2:
            keys.append("num:" + d)

    addr_words = [w for w in c_addr.split() if not w.isdigit() and len(w) >= 4 and w not in ADDR_STOPWORDS]
    for aw in addr_words[:4]:
        keys.append("aw:" + aw)

    if digits and addr_words:
        for d in digits[:2]:
            for aw in addr_words[:2]:
                keys.append("da:" + d + "_" + aw[:4])

    return keys


# -------------------------------------------------------------
# 4. TRAINING & PRECISION THRESHOLD TUNING
# -------------------------------------------------------------

def train_and_optimize():
    print("=" * 65)
    print("UPGRADED XGBOOST ER TRAINING PIPELINE (17 FEATURES)")
    print("=" * 65)

    s1_df = pl.read_csv(VAL_DIR / "val_source1.tsv", separator="\t")
    s2_df = pl.read_csv(VAL_DIR / "val_source2.tsv", separator="\t")
    s3_df = pl.read_csv(VAL_DIR / "val_source3.tsv", separator="\t")
    gt_df = pl.read_csv(VAL_DIR / "val_ground_truth.tsv", separator="\t")

    gt_map = {}
    for row in gt_df.iter_rows(named=True):
        m_str = row["matched_entity_ids"]
        if m_str and str(m_str).strip() != "" and str(m_str) != "nan":
            gt_map[row["source1_entity_id"]] = set(str(m_str).split(","))
        else:
            gt_map[row["source1_entity_id"]] = set()

    print(f"Loaded {len(s1_df):,} S1 validation records & ground truth.")

    # Pre-clean S1
    s1_raw_map = {}
    s1_clean_map = {}
    s1_keys_map = {}
    for row in s1_df.iter_rows(named=True):
        sid = row["entity_id"]
        bn = str(row["business_name"] or "")
        ba = str(row["business_address"] or "")
        s1_raw_map[sid] = (bn, ba)
        cn = clean_name(bn)
        ca = clean_address(ba)
        cd = set(extract_digits(ca))
        s1_clean_map[sid] = (cn, ca, cd)
        s1_keys_map[sid] = get_enhanced_keys(cn, ca)

    # Build Training Candidate Pairs from S2 and S3
    X_all = []
    y_all = []
    pair_meta = []

    for src_name, src_df in [("S2", s2_df), ("S3", s3_df)]:
        is_s2_val = 1.0 if src_name == "S2" else 0.0
        inv_index = defaultdict(list)
        tgt_records = []
        tgt_raw_list = []

        for i, row in enumerate(src_df.iter_rows(named=True)):
            tid = row["entity_id"]
            bn = str(row["business_name"] or "")
            ba = str(row["business_address"] or "")
            tgt_raw_list.append((bn, ba))
            cn = clean_name(bn)
            ca = clean_address(ba)
            tgt_records.append((tid, cn, ca))
            for k in get_enhanced_keys(cn, ca):
                inv_index[k].append(i)

        print(f"  Generating training candidate pairs for {src_name}...")
        for sid, (s1_cn, s1_ca, s1_cd) in s1_clean_map.items():
            s1_keys = s1_keys_map[sid]
            true_mids = gt_map.get(sid, set())

            cand_indices = set()
            for k in s1_keys:
                bucket = inv_index.get(k)
                if bucket and len(bucket) <= 2500:
                    cand_indices.update(bucket)
            if not cand_indices:
                continue

            scored_cands = []
            for tidx in cand_indices:
                tid, tcn, tca = tgt_records[tidx]
                sim = fuzz.token_sort_ratio(s1_cn, tcn)
                if s1_ca and tca:
                    sim = max(sim, fuzz.token_set_ratio(s1_ca, tca) * 0.95)
                if sim >= 45:
                    scored_cands.append((sim, tidx))

            scored_cands.sort(key=lambda x: x[0], reverse=True)
            s1_raw = s1_raw_map[sid]

            for sim, tidx in scored_cands[:15]:
                tid, tcn, tca = tgt_records[tidx]
                tgt_raw = tgt_raw_list[tidx]
                tcd = set(extract_digits(tca))
                feat = build_advanced_feature_vector(
                    s1_raw, tgt_raw, (s1_cn, s1_ca, s1_cd), (tcn, tca, tcd), is_s2_val
                )
                label = 1.0 if tid in true_mids else 0.0

                X_all.append(feat)
                y_all.append(label)
                pair_meta.append((sid, tid))

    X_mat = np.array(X_all, dtype=np.float32)
    y_arr = np.array(y_all, dtype=np.float32)
    print(f"Total Training Pairs: {len(y_arr):,} (Positives: {int(y_arr.sum()):,}, Negatives: {int(len(y_arr)-y_arr.sum()):,})")

    # Train XGBoost Classifier
    print("\nTraining Upgraded XGBoost Classifier...")
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"  • Accelerator: {device.upper()}")

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_mat, y_arr, test_size=0.25, random_state=42, stratify=y_arr
    )

    model = xgb.XGBClassifier(
        n_estimators=350,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        device=device,
        random_state=42,
        eval_metric="logloss",
    )

    t0 = time.time()
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    print(f"  • Training completed in {time.time() - t0:.2f} seconds!")

    # Switch to CPU multi-threading for inference & threshold tuning
    model.set_params(device="cpu", n_jobs=4)
    preds = model.predict_proba(X_mat)[:, 1]
    cand_probs = defaultdict(list)
    for idx, (sid, tid) in enumerate(pair_meta):
        cand_probs[sid].append((tid, float(preds[idx])))

    # Optimize Decision Threshold for Macro F_0.5
    print("\nTHRESHOLD OPTIMIZATION FOR MACRO F_0.5")
    best_thresh = 0.65
    best_f05 = 0.0

    for thresh in np.linspace(0.50, 0.85, 15):
        pred_map = {}
        for sid in gt_map.keys():
            cand_list = cand_probs.get(sid, [])
            filtered = {tid for tid, p in cand_list if p >= thresh}
            pred_map[sid] = filtered

        score = evaluate_macro_f05(gt_map, pred_map)
        print(f"  Threshold {thresh:.2f}  -->  Macro F_0.5 = {score:.4f} ({score*100:.2f}%)")
        if score > best_f05:
            best_f05 = score
            best_thresh = float(thresh)

    print("\n" + "=" * 65)
    print(f"OPTIMAL MACRO F_0.5 SCORE: {best_f05:.4f} ({best_f05*100:.2f}%) at Threshold {best_thresh:.2f}")
    print("=" * 65)

    # Feature Importance Breakdown
    print("\nFeature Importances:")
    importances = model.feature_importances_
    sorted_idx = np.argsort(-importances)
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:2d}. {FEATURE_NAMES[idx]:<22} : {importances[idx]:.4f}")

    # Save Serialized Model and Config
    model_file = MODELS_DIR / "xgboost_er_model.json"
    meta_file = MODELS_DIR / "model_config.json"
    model.save_model(str(model_file))

    config_data = {
        "best_threshold": best_thresh,
        "best_macro_f05": best_f05,
        "feature_names": FEATURE_NAMES,
    }
    with open(meta_file, "w") as f:
        json.dump(config_data, f, indent=2)

    # Sync to code/business_entity_resolution/models
    pkg_models = CURRENT_DIR.parent / "business_entity_resolution" / "models"
    pkg_models.mkdir(parents=True, exist_ok=True)
    model.save_model(str(pkg_models / "xgboost_er_model.json"))
    with open(pkg_models / "model_config.json", "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\nModel and configuration saved to {model_file} and synced to {pkg_models}!")


if __name__ == "__main__":
    train_and_optimize()
