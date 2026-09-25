

from collections import defaultdict
import os
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Add current folder to path for imports
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
try:
    from validation import clean_name, clean_address
except ImportError:
    import re
    def clean_name(t):
        return re.sub(r"[^\w\s]", " ", str(t or "")).lower().strip()
    def clean_address(t):
        return re.sub(r"[^\w\s]", " ", str(t or "")).lower().strip()

# Base Directories
BASE_DIR = CURRENT_DIR.parent.parent
DATA_DIR = BASE_DIR / "dataset"
VAL_DIR = DATA_DIR / "val"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class DualPassBlocker:
    """
    Dual-pass blocking engine:
      - Pass 1: Character 3-gram TF-IDF on Business Name (Top 15)
      - Pass 2: Word/Char TF-IDF on Business Address (Top 10)
    """

    def __init__(self, top_k_name: int = 15, top_k_addr: int = 10, total_k: int = 25):
        self.top_k_name = top_k_name
        self.top_k_addr = top_k_addr
        self.total_k = total_k

    def _block_feature(
        self,
        query_texts: list[str],
        target_texts: list[str],
        top_k: int,
        analyzer: str,
        ngram_range: tuple,
        min_sim: float = 0.12,
        batch_size: int = 2500,
    ) -> list[list[int]]:
        """Generic fast TF-IDF top-k retrieval returning list of target index lists."""
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=ngram_range,
            min_df=2,
            max_features=60000,
            dtype=np.float32,
        )
        target_tfidf = vectorizer.fit_transform(target_texts)
        query_tfidf = vectorizer.transform(query_texts)
        target_tfidf_T = target_tfidf.T.tocsc()

        total_queries = len(query_texts)
        results = []

        for start_idx in range(0, total_queries, batch_size):
            end_idx = min(start_idx + batch_size, total_queries)
            batch_query = query_tfidf[start_idx:end_idx]
            sim_batch = (batch_query @ target_tfidf_T).toarray()

            for i in range(end_idx - start_idx):
                sims = sim_batch[i]
                if len(sims) > top_k:
                    top_idx = np.argpartition(-sims, top_k)[:top_k]
                    top_idx = top_idx[np.argsort(-sims[top_idx])]
                else:
                    top_idx = np.argsort(-sims)

                valid_idx = [idx for idx in top_idx if sims[idx] >= min_sim]
                results.append(valid_idx)

        return results

    def block_country(
        self,
        s1_df: pl.DataFrame,
        targets_df: pl.DataFrame,
    ) -> dict[str, list[str]]:
        if len(s1_df) == 0 or len(targets_df) == 0:
            return {row["entity_id"]: [] for row in s1_df.iter_rows(named=True)}

        s1_ids = s1_df["entity_id"].to_list()
        target_ids = np.array(targets_df["entity_id"].to_list())

        print(f"    Cleaning {len(targets_df):,} targets and {len(s1_df):,} S1 queries...")
        s1_names = [clean_name(n) for n in s1_df["business_name"].to_list()]
        s1_addrs = [clean_address(a) for a in s1_df["business_address"].to_list()]

        target_names = [clean_name(n) for n in targets_df["business_name"].to_list()]
        target_addrs = [clean_address(a) for a in targets_df["business_address"].to_list()]

        # Pass 1: Name Blocking (Character 3-gram)
        print(f"    [Pass 1/2] Name Blocking (char 3-grams, top {self.top_k_name})...")
        name_results = self._block_feature(
            query_texts=s1_names,
            target_texts=target_names,
            top_k=self.top_k_name,
            analyzer="char_wb",
            ngram_range=(3, 3),
            min_sim=0.15,
        )

        # Pass 2: Address Blocking (Char 4-grams for street numbers & words)
        print(f"    [Pass 2/2] Address Blocking (char 4-grams, top {self.top_k_addr})...")
        addr_results = self._block_feature(
            query_texts=s1_addrs,
            target_texts=target_addrs,
            top_k=self.top_k_addr,
            analyzer="char_wb",
            ngram_range=(4, 4),
            min_sim=0.15,
        )

        # Merge results preserving order and removing duplicates
        print("    Merging Name + Address candidates...")
        candidates_map = {}
        for i, s1_id in enumerate(s1_ids):
            name_cands = [target_ids[idx] for idx in name_results[i]]
            addr_cands = [target_ids[idx] for idx in addr_results[i]]

            # Combine without duplicates
            seen = set()
            merged = []
            for cid in name_cands:
                if cid not in seen:
                    seen.add(cid)
                    merged.append(cid)
            for cid in addr_cands:
                if cid not in seen and len(merged) < self.total_k:
                    seen.add(cid)
                    merged.append(cid)

            candidates_map[s1_id] = merged

        return candidates_map


def evaluate_blocking_recall(candidates_map: dict[str, list[str]], gt_path: Path):
    print("\n" + "=" * 60)
    print("DUAL-PASS BLOCKING EVALUATION REPORT")
    print("=" * 60)
    gt_df = pl.read_csv(gt_path, separator="\t")
    total_true_matches = 0
    captured_matches = 0
    candidate_counts = []

    for row in gt_df.iter_rows(named=True):
        s1_id = row["source1_entity_id"]
        m_str = row["matched_entity_ids"]
        cands = set(candidates_map.get(s1_id, []))
        candidate_counts.append(len(cands))

        if m_str and str(m_str).strip() != "" and str(m_str) != "nan":
            true_ids = set(str(m_str).split(","))
            total_true_matches += len(true_ids)
            captured = len(true_ids & cands)
            captured_matches += captured

    recall_ceiling = (captured_matches / total_true_matches) * 100 if total_true_matches > 0 else 0
    avg_cands = np.mean(candidate_counts) if candidate_counts else 0

    print(f"  • Total True Matches in Ground Truth:   {total_true_matches:,}")
    print(f"  • True Matches Captured by Blocking:   {captured_matches:,}")
    print(f"  • RECALL CEILING:                      {recall_ceiling:.2f}%")
    print(f"  • Avg Candidates per S1 Entity:        {avg_cands:.1f}")
    print("=" * 60)


def run_blocking(data_split_dir: Path, output_file: Path):
    start_time = time.time()
    print("=" * 60)
    print(f"Running Dual-Pass Blocking on dataset: {data_split_dir}")
    print("=" * 60)

    prefix = "val_" if "val" in str(data_split_dir) else "test_"
    s1_file = data_split_dir / f"{prefix}source1.tsv"
    s2_file = data_split_dir / f"{prefix}source2.tsv"
    s3_file = data_split_dir / f"{prefix}source3.tsv"

    print("Loading data files...")
    s1_df = pl.read_csv(s1_file, separator="\t")
    s2_df = pl.read_csv(s2_file, separator="\t")
    s3_df = pl.read_csv(s3_file, separator="\t")
    targets_df = pl.concat([s2_df, s3_df])

    print(f"  Loaded {len(s1_df):,} S1 entities and {len(targets_df):,} Target (S2+S3) records.")
    countries = s1_df["country"].unique().to_list()

    blocker = DualPassBlocker(top_k_name=15, top_k_addr=10, total_k=25)
    all_candidates: dict[str, list[str]] = {}

    for country in countries:
        print(f"\n--> Processing Country: {country}")
        s1_country = s1_df.filter(pl.col("country") == country)
        targets_country = targets_df.filter(pl.col("country") == country)
        print(f"    S1: {len(s1_country):,}, Targets: {len(targets_country):,}")

        cands = blocker.block_country(s1_country, targets_country)
        all_candidates.update(cands)

    # Save to TSV
    print(f"\nWriting candidate pairs to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_df["entity_id"].to_list():
            cand_str = ",".join(all_candidates.get(s1_id, []))
            f.write(f"{s1_id}\t{cand_str}\n")

    elapsed = time.time() - start_time
    print(f"Blocking complete in {elapsed:.1f}s.")

    gt_file = data_split_dir / f"{prefix}ground_truth.tsv"
    if gt_file.exists():
        evaluate_blocking_recall(all_candidates, gt_file)


if __name__ == "__main__":
    val_output = OUTPUT_DIR / "candidate_pairs.tsv"
    run_blocking(VAL_DIR, val_output)
