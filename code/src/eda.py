from collections import Counter
import os
from pathlib import Path
import polars as pl

# Automatically find student_resource root directory from this file's path
# eda.py is in student_resource/code/src/eda.py -> parents[2] is student_resource
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Accessing the train data
TRAIN_S1 = str(TRAIN_DIR / "train_source1.tsv")
TRAIN_S2 = str(TRAIN_DIR / "train_source2.tsv")
TRAIN_S3 = str(TRAIN_DIR / "train_source3.tsv")
TRAIN_GT = str(TRAIN_DIR / "train_ground_truth.tsv")

# Accessing the test data
TEST_S1 = str(TEST_DIR / "test_source1.tsv")
TEST_S2 = str(TEST_DIR / "test_source2.tsv")
TEST_S3 = str(TEST_DIR / "test_source3.tsv")

def analyze_gd_truth():
    print("=" * 60)
    print("1. Ground Truth & Singleton Analysis")
    print("=" * 60)

    #Read ground truth data with polars
    gt_df = pl.read_csv(TRAIN_GT, separator="\t")
    total_s1 = len(gt_df)
    print(f"Total Source 1 records in Training: {total_s1:,}")

    #Process match lists
    #Note: null or empty string means 0 matches(singleton)
    matches_raw = gt_df["matched_entity_ids"].to_list()

    match_counts = []
    source_counts = Counter()
    singletons = 0

    for m in matches_raw:
        if m is None or str(m).strip() == "" or str(m) == "nan":
            singletons += 1
            match_counts.append(0)
        else:
            ids = [x.strip() for x in str(m).split(",") if x.strip()]
            match_counts.append(len(ids))
            for mid in ids:
                if mid.startswith("S2-"):
                    source_counts["S2"] += 1
                elif mid.startswith("S3-"):
                    source_counts["S3"] += 1

    singleton_pct = (singletons/total_s1) * 100
    matched_pct = 100 - singleton_pct

    print((f"Singletons (0 matches): {singletons:,} ({singleton_pct:.2f}%)"))
    print(f"Entity With matches: {total_s1 - singletons:,} ({matched_pct:.2f}%)" )
    print(f"Total S2 matched reference: {source_counts['S2']:,}")
    print(f"Total S3 matched reference: {source_counts['S3']:,}")
    
    # Now match the count Distribution
    cnt_dist = Counter(match_counts)
    print("\nMatch count distribution per s1 entity:")
    for count in sorted(cnt_dist.keys())[:7]:
        pct = (cnt_dist[count] / total_s1) * 100
        print(f"Matches = {count}: {cnt_dist[count]:,} entities ({pct:.2f}%)")

    if any(k > 6 for k in cnt_dist.keys()):
        greater = sum(v for k, v in cnt_dist.items() if k > 6)
        print(f" matches > 6: {greater:,} entities ({(greater/total_s1) * 100:.2f}%)")

def analyze_country_distribution():

    print("\n" + "=" * 60)
    print("2. COUNTRY DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Training S1 countries
    train_s1_df = pl.read_csv(TRAIN_S1, separator="\t", columns=["country"])
    train_counts = train_s1_df["country"].value_counts().sort("count", descending=True)
    print("Training Set (Source 1):")
    for row in train_counts.iter_rows():
        pct = (row[1] / len(train_s1_df)) * 100
        print(f"  {row[0]}: {row[1]:,} ({pct:.2f}%)")

    # Test S1 countries
    test_s1_df = pl.read_csv(TEST_S1, separator="\t", columns=["country"])
    test_counts = test_s1_df["country"].value_counts().sort("count", descending=True)
    print("\nTest Set (Source 1 - includes France):")

    for row in test_counts.iter_rows():
        pct = (row[1] / len(test_s1_df)) * 100
        print(f"  {row[0]}: {row[1]:,} ({pct:.2f}%)")
    

def inspect_matching_examples(sample_size=3):
    print("\n" + "=" * 60)
    print("3. SIDE-BY-SIDE MATCHING EXAMPLES (Noise & Variation)")
    print("=" * 60)
    gt_df = pl.read_csv(TRAIN_GT, separator="\t")
    # Filter for entities that match both S2 and S3
    sample_entities = []
    for row in gt_df.iter_rows(named=True):
        m = row["matched_entity_ids"]
        if m and "S2-" in m and "S3-" in m:
            sample_entities.append(row)
            if len(sample_entities) >= sample_size:
                break
    target_s1_ids = {r["source1_entity_id"] for r in sample_entities}
    all_target_ids = set(target_s1_ids)
    for r in sample_entities:
        for mid in r["matched_entity_ids"].split(","):
            all_target_ids.add(mid.strip())
    # Load only necessary rows
    s1_sample = (
        pl.read_csv(TRAIN_S1, separator="\t")
        .filter(pl.col("entity_id").is_in(list(target_s1_ids)))
        .to_dicts()
    )
    s1_map = {row["entity_id"]: row for row in s1_sample}
    s2_sample = (
        pl.read_csv(TRAIN_S2, separator="\t")
        .filter(pl.col("entity_id").is_in(list(all_target_ids)))
        .to_dicts()
    )
    s2_map = {row["entity_id"]: row for row in s2_sample}
    s3_sample = (
        pl.read_csv(TRAIN_S3, separator="\t")
        .filter(pl.col("entity_id").is_in(list(all_target_ids)))
        .to_dicts()
    )
    s3_map = {row["entity_id"]: row for row in s3_sample}
    for idx, entity in enumerate(sample_entities, 1):
        s1_id = entity["source1_entity_id"]
        s1_data = s1_map.get(s1_id, {})
        print(f"\n--- [Example {idx}] Reference Entity: {s1_id} ({s1_data.get('country')}) ---")
        print(f"  [S1 Reference]")
        print(f"    Name:    {s1_data.get('business_name')}")
        print(f"    Address: {s1_data.get('business_address')}")
        matched_ids = [x.strip() for x in entity["matched_entity_ids"].split(",") if x.strip()]
        for mid in matched_ids:
            if mid.startswith("S2-") and mid in s2_map:
                d = s2_map[mid]
                print(f"  [Matched {mid}]")
                print(f"    Name:    {d.get('business_name')}")
                print(f"    Address: {d.get('business_address')}")
            elif mid.startswith("S3-") and mid in s3_map:
                d = s3_map[mid]
                print(f"  [Matched {mid}]")
                print(f"    Name:    {d.get('business_name')}")
                print(f"    Address: {d.get('business_address')}")


if __name__ == "__main__":
    analyze_gd_truth()
    analyze_country_distribution()
    inspect_matching_examples()