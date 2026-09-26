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
    from train_model import build_advanced_feature_vector
except ImportError:
    from code.src.validation import (
        clean_address,
        clean_name,
        extract_digits,
        extract_pin,
        extract_primary_bldg,
        is_non_latin,
    )
    from code.src.train_model import build_advanced_feature_vector

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
    "international", "holding", "holdings", "management", "india", "france", "us",
    "center", "centre", "com", "partners", "associates", "consulting"
}

ADDR_STOPWORDS = {
    "road", "street", "avenue", "lane", "drive", "court", "floor", "apartment",
    "near", "opposite", "cross", "main", "nagar", "city", "state", "district",
    "rue", "boulevard", "place", "allée", "chemin", "route", "plot", "block",
    "door", "west", "east", "north", "south", "null", "bldg", "building"
}

RE_DIGITS = re.compile(r"\b\d+\b")


def get_blocking_keys(c_name: str, c_addr: str) -> list[str]:
    """
    Generates high-precision multi-aspect blocking keys without generic noise:
      1. Compact Name Prefix (5, 4, 3 chars)
      2. Significant Name Words (distinctive words)
      3. Compound: PIN + Name Prefix (ultra-high precision)
      4. Compound: Building Number + Street Word (e.g. da:5_pier)
      5. Compound: PIN + Street Word
      6. Cross-script Indic Compound: PIN + Building Number
    """
    keys = []
    compact = c_name.replace(" ", "")
    if len(compact) >= 5:
        keys.append("p5:" + compact[:5])
        keys.append("p4:" + compact[:4])
    elif len(compact) >= 4:
        keys.append("p4:" + compact[:4])
        keys.append("p3:" + compact[:3])
    elif len(compact) >= 3:
        keys.append("p3:" + compact[:3])

    words = [w for w in c_name.split() if len(w) >= 3 and w not in LEGAL_STOPWORDS]
    for w in words[:4]:
        keys.append("w:" + w)

    pin = extract_pin(c_addr)
    if pin and len(compact) >= 3:
        keys.append("pin_p:" + pin + "_" + compact[:3])

    digits = [d for d in extract_digits(c_addr) if len(d) <= 6]
    addr_words = [w for w in c_addr.split() if not w.isdigit() and len(w) >= 4 and w not in ADDR_STOPWORDS]

    if digits and addr_words:
        for d in digits[:2]:
            for aw in addr_words[:2]:
                keys.append("da:" + d + "_" + aw[:4])

    if pin and addr_words:
        for aw in addr_words[:2]:
            keys.append("pa:" + pin + "_" + aw[:4])

    if is_non_latin(c_name) or (pin and digits):
        if pin and digits:
            for d in digits[:2]:
                if d != pin:
                    keys.append("pd:" + pin + "_" + d)

    return keys


def stream_s1_for_country(file_path: Path, target_country: str) -> list[tuple]:
    """
    Reads S1 records line-by-line for a specific country with minimal RAM footprint.
    Pre-extracts regex features (PIN, building, non-latin) once to avoid nested regex overhead.
    Returns: list of (entity_id, raw_record, clean_record_with_cached_regex, blocking_keys)
    """
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and parts[3] == target_country:
                eid = parts[0]
                raw_name = parts[1]
                raw_addr = parts[2]
                cn = clean_name(raw_name)
                ca = clean_address(raw_addr)
                cd = set(extract_digits(ca))
                pin = extract_pin(raw_addr)
                bldg = extract_primary_bldg(raw_addr)
                is_nl = 1.0 if is_non_latin(raw_name) else 0.0
                keys = get_blocking_keys(cn, ca)
                records.append((eid, (raw_name, raw_addr), (cn, ca, cd, pin, bldg, is_nl), keys))
    return records


def stream_targets_and_index(file_path: Path, target_country: str) -> tuple[list[tuple], dict[str, list[int]]]:
    """
    Streams target records (S2 or S3) line-by-line and constructs an integer inverted index.
    Pre-extracts regex features once during load for 65x faster candidate evaluation.
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
                raw_name = parts[1]
                raw_addr = parts[2]
                cn = clean_name(raw_name)
                ca = clean_address(raw_addr)
                cd = set(extract_digits(ca))
                pin = extract_pin(raw_addr)
                bldg = extract_primary_bldg(raw_addr)
                is_nl = 1.0 if is_non_latin(raw_name) else 0.0
                records.append((eid, (raw_name, raw_addr), (cn, ca, cd, pin, bldg, is_nl)))
                for k in get_blocking_keys(cn, ca):
                    inv_index[k].append(idx)
                idx += 1
    return records, inv_index


def run_lean_inference():
    start_total = time.time()
    print("=" * 68, flush=True)
    print("GPU-ACCELERATED HIGH-PERFORMANCE INFERENCE (RAM < 1.1 GB)", flush=True)
    print("=" * 68, flush=True)

    # 1. Load Trained XGBoost Model & Optimal Decision Threshold
    model_file = MODELS_DIR / "xgboost_er_model.json"
    meta_file = MODELS_DIR / "model_config.json"

    if not model_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Model missing at {model_file}! Train the model first.")

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"

    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    if use_cuda:
        model.set_params(device="cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Inference Engine: NVIDIA GPU Acceleration via CUDA ({gpu_name})", flush=True)
    else:
        model.set_params(device="cpu", n_jobs=4)
        print("Inference Engine: Multi-Threaded CPU XGBoost (4 threads)", flush=True)

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
    batch_size = 50000 if use_cuda else 25000

    for country in countries:
        country_start = time.time()
        print("\n" + "=" * 68, flush=True)
        print(f"PROCESSING COUNTRY: {country.upper()}", flush=True)
        print("=" * 68, flush=True)

        # 1. Load S1 queries for this country
        t0 = time.time()
        print(f"  Streaming S1 entities for {country}...", flush=True)
        s1_records = stream_s1_for_country(s1_path, country)
        n_s1 = len(s1_records)
        print(f"  Loaded {n_s1:,} S1 entities in {time.time() - t0:.1f}s.", flush=True)

        # Candidate pool and scored match candidates per S1 entity
        candidates_by_s1 = defaultdict(list)
        scored_matches_by_s1 = defaultdict(list)

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
            print(f"  Generating candidates & scoring for {src_label} on GPU...", flush=True)
            s1_chunk_pairs = []
            s1_chunk_feats = []
            s1_chunk_meta = []
            t_start_loop = time.time()

            for i, (sid, s1_raw, s1_clean_tuple, s1_keys) in enumerate(s1_records):
                s1_cn, s1_ca = s1_clean_tuple[0], s1_clean_tuple[1]

                # Aggregate candidate indices from all matching keys
                cand_indices = set()
                for k in s1_keys:
                    bucket = inv_index.get(k)
                    if bucket and len(bucket) <= 1500:  # Skip high-frequency noise keys
                        cand_indices.update(bucket)

                if not cand_indices:
                    continue

                # Pre-rank candidate targets using RapidFuzz quick similarity
                scored_cands = []
                for tidx in cand_indices:
                    tid, tgt_raw, tgt_clean_tuple = tgt_records[tidx]
                    tcn, tca = tgt_clean_tuple[0], tgt_clean_tuple[1]
                    sim = fuzz.token_sort_ratio(s1_cn, tcn)
                    if s1_ca and tca:
                        a_sim = fuzz.token_set_ratio(s1_ca, tca)
                        sim = max(sim, a_sim * 0.95)
                    # Filter candidates with plausible match signal
                    if sim >= 45:
                        scored_cands.append((sim, tidx))

                scored_cands.sort(key=lambda x: x[0], reverse=True)

                # Keep top 15 highest-similarity candidates for this source
                for sim, tidx in scored_cands[:15]:
                    tid, tgt_raw, tgt_clean_tuple = tgt_records[tidx]
                    candidates_by_s1[sid].append(tid)

                    feat = build_advanced_feature_vector(
                        s1_raw, tgt_raw, s1_clean_tuple, tgt_clean_tuple, is_s2_val
                    )
                    s1_chunk_feats.append(feat)
                    s1_chunk_pairs.append((sid, tid))
                    s1_chunk_meta.append((s1_raw, tgt_raw, feat))

                # Batch score on GPU in milliseconds when buffer reaches threshold
                if len(s1_chunk_feats) >= batch_size:
                    X_mat = np.array(s1_chunk_feats, dtype=np.float32)
                    probs = model.predict_proba(X_mat)[:, 1]
                    for p_idx, (q_sid, q_tid) in enumerate(s1_chunk_pairs):
                        scored_matches_by_s1[q_sid].append((q_tid, float(probs[p_idx]), s1_chunk_meta[p_idx]))

                    s1_chunk_pairs.clear()
                    s1_chunk_feats.clear()
                    s1_chunk_meta.clear()

                if (i + 1) % 50000 == 0 or (i + 1) == n_s1:
                    pct = ((i + 1) / n_s1) * 100
                    elapsed = time.time() - t_start_loop
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    print(f"    [{src_label}] Processed {i + 1:,}/{n_s1:,} queries ({pct:.1f}%) at {rate:.1f} queries/sec...", flush=True)

            # Flush any remaining buffer pairs on GPU
            if s1_chunk_feats:
                X_mat = np.array(s1_chunk_feats, dtype=np.float32)
                probs = model.predict_proba(X_mat)[:, 1]
                for p_idx, (q_sid, q_tid) in enumerate(s1_chunk_pairs):
                    scored_matches_by_s1[q_sid].append((q_tid, float(probs[p_idx]), s1_chunk_meta[p_idx]))
                s1_chunk_pairs.clear()
                s1_chunk_feats.clear()
                s1_chunk_meta.clear()

            # Clean up target memory immediately before loading next source
            del tgt_records, inv_index
            gc.collect()

        # 3. Apply Precision Guards & Relative Margin Pruning, then Stream to Disk
        print(f"\n  Applying precision guards & writing {n_s1:,} {country} results to disk...", flush=True)
        with open(cand_out_path, "a", encoding="utf-8") as f_c, \
             open(match_out_path, "a", encoding="utf-8") as f_m:

            for sid, s1_raw, s1_clean_tuple, _ in s1_records:
                # 1. Candidate Pairs Output (Preserve all generated candidates)
                c_list = list(dict.fromkeys(candidates_by_s1.get(sid, [])))
                f_c.write(f"{sid}\t{','.join(c_list)}\n")

                # 2. Precision-Guarded Match Selection
                raw_cands = scored_matches_by_s1.get(sid, [])
                filtered_cands = []

                for tid, p, (s1_r, tgt_r, feat) in raw_cands:
                    postal_status = feat[11]  # postal_match_status
                    bldg_status = feat[12]    # bldg_match_status
                    is_cross = feat[7]        # is_cross_script
                    name_sort = feat[0]
                    name_set = feat[1]
                    name_jw = feat[2]
                    core_sim = feat[5]

                    # Veto 1: Postal Code Conflict (different city / postal code)
                    if postal_status == -1.0:
                        continue

                    # Veto 2: Building Number Mismatch (different door number on same street)
                    if bldg_status == -1.0 and name_sort < 0.85:
                        continue

                    # Veto 3: Co-located Latin Name Consistency Guard (prevent merging distinct shops at same address)
                    if is_cross == 0.0:
                        c1 = clean_name(s1_r[0]).replace(" ", "")
                        c2 = clean_name(tgt_r[0]).replace(" ", "")
                        contained = (len(c1) >= 3 and c1[:4] in c2) or (len(c2) >= 3 and c2[:4] in c1)
                        # Acronym check (e.g. Swastik High Media -> SHM)
                        w1 = [w[0] for w in s1_r[0].split() if w]
                        w2 = [w[0] for w in tgt_r[0].split() if w]
                        acronym_match = ("".join(w1).lower() == c2.lower()) or ("".join(w2).lower() == c1.lower())

                        if max(name_sort, name_set, core_sim) < 0.35 and name_jw < 0.60 and not contained and not acronym_match:
                            continue

                    if p >= best_thresh:
                        filtered_cands.append((tid, p))

                # Adaptive Relative Margin Pruning & Top-6 Match Cap
                if filtered_cands:
                    filtered_cands.sort(key=lambda x: x[1], reverse=True)
                    max_p = filtered_cands[0][1]
                    # Prune low-confidence stragglers far below the top candidate
                    pruned_cands = [tid for tid, p in filtered_cands if p >= (max_p - 0.22)][:6]
                    m_list = list(dict.fromkeys(pruned_cands))
                else:
                    m_list = []

                f_m.write(f"{sid}\t{','.join(m_list)}\n")

        country_duration = time.time() - country_start
        total_processed += n_s1
        print(f"  Completed {country} in {country_duration / 60:.2f} minutes.", flush=True)
        print(f"  Progress: {total_processed:,}/{1732544:,} total S1 entities written.", flush=True)

        # Completely free country memory
        del s1_records, candidates_by_s1, scored_matches_by_s1
        gc.collect()

    total_time = time.time() - start_total
    print("\n" + "=" * 68, flush=True)
    print(f"ALL TEST INFERENCE COMPLETED IN {total_time / 60:.2f} MINUTES!", flush=True)
    print(f"  Candidate Pairs:  {cand_out_path}", flush=True)
    print(f"  Matching Results: {match_out_path}", flush=True)
    print("=" * 68, flush=True)


if __name__ == "__main__":
    run_lean_inference()
