

from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from rapidfuzz import fuzz, distance
import xgboost as xgb

# Path setup
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
try:
    from validation import clean_address, clean_name, extract_digits
    from train_model import build_feature_vector
except ImportError:
    from code.src.validation import clean_address, clean_name, extract_digits
    from code.src.train_model import build_feature_vector

BASE_DIR = CURRENT_DIR.parent.parent
DATA_DIR = BASE_DIR / "dataset"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Common corporate / legal stopwords to avoid indexing giant noise buckets
LEGAL_STOPWORDS = {
    "pvt", "ltd", "limited", "private", "llc", "corp", "corporation", "inc",
    "incorporated", "co", "company", "sarl", "sa", "sas", "enterprises",
    "solutions", "services", "the", "and", "for", "group", "industries",
    "international", "holding", "holdings", "management", "india", "france", "us"
}

RE_DIGITS = re.compile(r"\b\d+\b")


def get_blocking_keys(c_name: str, c_addr: str) -> list[str]:
    """
    Generates multi-aspect blocking keys:
      1. Compact Name Prefix (5 chars and 4 chars)
      2. Significant Name Words (up to 5 words)
      3. Postal code (5 or 6 digits)
      4. Building Number + Street Root
      5. Building Number alone
    """
    keys = []
    compact = c_name.replace(" ", "")
    if len(compact) >= 4:
        keys.append("p:" + compact[:5])
        keys.append("p4:" + compact[:4])

    words = [w for w in c_name.split() if len(w) >= 3 and w not in LEGAL_STOPWORDS]
    for w in words[:5]:
        keys.append("w:" + w)

    digits = RE_DIGITS.findall(c_addr)
    # Postal codes (5-digit US or 6-digit India)
    for d in digits:
        if len(d) in (5, 6):
            keys.append("pin:" + d)

    addr_words = [w for w in c_addr.split() if not w.isdigit() and len(w) >= 3]
    if digits and addr_words:
        keys.append("a:" + digits[0] + "_" + addr_words[0][:4])
    elif digits:
        keys.append("n:" + digits[0])

    return keys


def stream_s1_for_country(file_path: Path, target_country: str) -> list[tuple]:
    """
    Reads S1 records line-by-line for a specific country with minimal RAM footprint.
    Returns: list of (entity_id, cleaned_name, cleaned_addr, digit_set, blocking_keys)
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3] == target_country:
                eid = parts[0]
                cn = clean_name(parts[1])
                ca = clean_address(parts[2])
                cd = set(extract_digits(ca))
                keys = get_blocking_keys(cn, ca)
                records.append((eid, cn, ca, cd, keys))
    return records


def stream_targets_and_index(file_path: Path, target_country: str) -> tuple[list[tuple], dict[str, list[int]]]:
    """
    Streams target records (S2 or S3) line-by-line and constructs an integer inverted index.
    RAM is strictly proportional only to target count (~250 MB for France, ~750 MB for India).
    """
    records = []
    inv_index = defaultdict(list)
    idx = 0
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3] == target_country:
                eid = parts[0]
                cn = clean_name(parts[1])
                ca = clean_address(parts[2])
                records.append((eid, cn, ca))
                for k in get_blocking_keys(cn, ca):
                    inv_index[k].append(idx)
                idx += 1
    return records, inv_index


def run_lean_inference():
    start_total = time.time()
    print("LEAN & FAST TEST INFERENCE PIPELINE (Strict RAM < 850 MB)", flush=True)

    # 1. Load Trained XGBoost Model & Optimal Decision Threshold
    model_file = MODELS_DIR / "xgboost_er_model.json"
    meta_file = MODELS_DIR / "model_config.json"

    if not model_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Model missing at {model_file}! Train the model first.")

    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    # CPU multi-threaded prediction: zero GPU memory allocation, ~31ms per 25k batch
    model.set_params(device="cpu", n_jobs=4)
    print("Inference Engine: Multi-Threaded CPU XGBoost (4 threads, in-place)", flush=True)

    with open(meta_file, "r") as f:
        meta = json.load(f)
    best_thresh = meta.get("best_threshold", 0.65)
    print(f"Decision Threshold: {best_thresh:.2f}", flush=True)

    # 2. Prepare Output Files with Official Competition Headers
    cand_out_path = OUTPUT_DIR / "candidate_pairs.tsv"
    match_out_path = OUTPUT_DIR / "matching_results.tsv"

    with open(cand_out_path, "w", encoding="utf-8") as f_c:
        f_c.write("source1_entity_id\tcandidate_entity_ids\n")
    with open(match_out_path, "w", encoding="utf-8") as f_m:
        f_m.write("source1_entity_id\tmatched_entity_ids\n")

    s1_path = TEST_DIR / "test_source1.tsv"
    s2_path = TEST_DIR / "test_source2.tsv"
    s3_path = TEST_DIR / "test_source3.tsv"

    countries = ["France", "US", "India"]
    total_processed = 0

    for country in countries:
        country_start = time.time()
        print(f"PROCESSING COUNTRY: {country.upper()}", flush=True)

        # 1. Load S1 queries for this country
        t0 = time.time()
        print(f"  Streaming S1 entities for {country}...", flush=True)
        s1_records = stream_s1_for_country(s1_path, country)
        n_s1 = len(s1_records)
        print(f"  Loaded {n_s1:,} S1 entities in {time.time() - t0:.1f}s.", flush=True)

        # Candidate pool and match pool per S1 entity
        candidates_by_s1 = defaultdict(list)
        matches_by_s1 = defaultdict(list)

        # 2. Two-Pass Evaluation: Process S2 Targets, then S3 Targets separately
        target_sources = [
            ("Source 2", s2_path, 1.0),
            ("Source 3", s3_path, 0.0),
        ]

        for src_label, src_path, is_s2_val in target_sources:
            t_src = time.time()
            print(f"\n  --- Indexing {src_label} for {country} ---", flush=True)
            tgt_records, inv_index = stream_targets_and_index(src_path, country)
            n_tgt = len(tgt_records)
            print(f"  Indexed {n_tgt:,} {src_label} targets ({len(inv_index):,} keys) in {time.time() - t_src:.1f}s.", flush=True)

            # Stream S1 through candidate generation & batch scoring
            print(f"  Generating candidates & scoring for {src_label}...", flush=True)
            s1_chunk_pairs = []
            s1_chunk_feats = []

            for i, (sid, s1_cn, s1_ca, s1_cd, s1_keys) in enumerate(s1_records):
                # Aggregate candidate indices from all matching keys
                cand_indices = set()
                for k in s1_keys:
                    bucket = inv_index.get(k)
                    if bucket and len(bucket) <= 1200:  # Skip high-frequency noise keys
                        cand_indices.update(bucket)
                        if len(cand_indices) >= 300:
                            break

                if not cand_indices:
                    continue

                # Pre-rank candidate targets using RapidFuzz quick similarity
                scored_cands = []
                for tidx in list(cand_indices)[:250]:
                    tid, tcn, tca = tgt_records[tidx]
                    sim = fuzz.token_sort_ratio(s1_cn, tcn)
                    if sim < 50 and s1_ca and tca:
                        sim = max(sim, fuzz.token_set_ratio(s1_ca, tca) * 0.9)
                    scored_cands.append((sim, tidx))

                scored_cands.sort(key=lambda x: x[0], reverse=True)

                # Keep top 15 highest-similarity candidates for this source
                for sim, tidx in scored_cands[:15]:
                    tid, tcn, tca = tgt_records[tidx]
                    candidates_by_s1[sid].append(tid)

                    tcd = set(extract_digits(tca))
                    feat = build_feature_vector((s1_cn, s1_ca, s1_cd), (tcn, tca, tcd), is_s2_val)
                    s1_chunk_feats.append(feat)
                    s1_chunk_pairs.append((sid, tid))

                # Batch score on CPU when buffer reaches threshold
                if len(s1_chunk_feats) >= 25000:
                    X_mat = np.array(s1_chunk_feats, dtype=np.float32)
                    probs = model.predict_proba(X_mat)[:, 1]
                    for p_idx, (q_sid, q_tid) in enumerate(s1_chunk_pairs):
                        if probs[p_idx] >= best_thresh:
                            matches_by_s1[q_sid].append(q_tid)

                    s1_chunk_pairs.clear()
                    s1_chunk_feats.clear()

                if (i + 1) % 50000 == 0 or (i + 1) == n_s1:
                    pct = ((i + 1) / n_s1) * 100
                    print(f"    [{src_label}] Processed {i + 1:,}/{n_s1:,} queries ({pct:.1f}%)...", flush=True)

            # Flush any remaining buffer pairs
            if s1_chunk_feats:
                X_mat = np.array(s1_chunk_feats, dtype=np.float32)
                probs = model.predict_proba(X_mat)[:, 1]
                for p_idx, (q_sid, q_tid) in enumerate(s1_chunk_pairs):
                    if probs[p_idx] >= best_thresh:
                        matches_by_s1[q_sid].append(q_tid)
                s1_chunk_pairs.clear()
                s1_chunk_feats.clear()

            # Clean up target memory immediately before loading next source
            del tgt_records, inv_index
            gc.collect()

        # 3. Stream Results for this Country Directly to Output Files
        print(f"\n  Writing {n_s1:,} {country} results to disk...", flush=True)
        with open(cand_out_path, "a", encoding="utf-8") as f_c, \
             open(match_out_path, "a", encoding="utf-8") as f_m:

            for sid, _, _, _, _ in s1_records:
                c_list = list(dict.fromkeys(candidates_by_s1.get(sid, [])))
                f_c.write(f"{sid}\t{','.join(c_list)}\n")

                m_list = list(dict.fromkeys(matches_by_s1.get(sid, [])))
                f_m.write(f"{sid}\t{','.join(m_list)}\n")

        country_duration = time.time() - country_start
        total_processed += n_s1
        print(f"  Completed {country} in {country_duration / 60:.2f} minutes.", flush=True)
        print(f"  Progress: {total_processed:,}/{1732544:,} total S1 entities written.", flush=True)

        # Completely free country memory
        del s1_records, candidates_by_s1, matches_by_s1
        gc.collect()

    total_time = time.time() - start_total
    print(f"ALL TEST INFERENCE COMPLETED IN {total_time / 60:.2f} MINUTES!", flush=True)
    print(f"  Candidate Pairs:  {cand_out_path}", flush=True)
    print(f"  Matching Results: {match_out_path}", flush=True)


if __name__ == "__main__":
    run_lean_inference()
