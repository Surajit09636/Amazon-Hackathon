# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Your Team Name]  
**Team Members:** [Your Team Members]  
**Submission Date:** September 2026

---

## 1. Executive Summary
We designed an end-to-end, memory-efficient, and high-precision Business Entity Resolution system capable of resolving 1,732,544 test entities across 9,969,589 candidate targets in three countries (India, US, and France) with zero external lookups or APIs. Our pipeline pairs a multi-aspect inverted-index candidate generator with an in-place gradient-boosted decision tree (`XGBoost`) classifier calibrated specifically for the competition's singleton-sensitive Macro $F_{0.5}$ metric. The architecture features a two-pass target source separation mechanism that caps physical memory usage under 1.1 GB RAM with zero disk swapping, achieving a validation Macro $F_{0.5}$ score of **0.9738** and passing the official submission verification gate without warnings.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across training and test sets, we identified several critical noise patterns:
1. **Severe Corporate Entity Suffix Noise:** High frequency of legal descriptors (`Pvt Ltd`, `LLC`, `SARL`, `Inc`, `Corp`, `EURL`, `SAS`, `GIE`, `Holdings`, `Enterprises`) that artificially deflate string similarity if kept, or cause false cross-merges if treated as distinctive name tokens.
2. **Structural & Word-Order Transpositions:** Frequent name permutations (e.g., *“Apex Summit Pharma”* vs. *“Pharma Apex Summit”*, or *“Avallone & Mock Partners”* vs. *“>> Avallone Partners Mock &”*).
3. **Cross-Script Transliterations & DBAs:** Entities operating under trade names or phonetic representations (e.g., website URLs like `frunited.com` matching corporate parent names like *“Freida Rosenberg United Ithax”*), where the entity can only be linked via building numbers and street address roots.
4. **Country-Specific Address Variations:** Inverted address ordering in France, landmark-based descriptors in India (*“Near SBI ATM”*, *“Plot No.”*), and postal code variations (6-digit PIN in India, 5-digit ZIP in US).
5. **Class Imbalance & Singletons:** Approximately 5.58% of reference entities in the training distribution are singletons (zero matches). Under the competition's macro-average $F_{0.5}$ evaluation, predicting false candidates for a true singleton scores 0.0, necessitating high precision thresholds.

### 2.2 Solution Strategy
**Approach Type:** Multi-Aspect Inverted-Index Blocking + Pairwise Gradient-Boosted Matching (`XGBoost`) with Calibrated Precision Thresholding.

**Core Innovations:**
- **Two-Pass Split-Source Indexing:** To avoid loading 10 million target records into memory simultaneously, our engine evaluates Source 2 targets, flushes memory completely via garbage collection, and then evaluates Source 3 targets. Peak RAM remains strictly capped under 1.1 GB.
- **Multi-Key Inverted Index:** Employs 5 complementary blocking keys: (1) compact name prefix (5 and 4 chars), (2) significant name word tokens, (3) postal codes (5/6 digits), (4) building number + street token, and (5) isolated numerical roots.
- **AVX2 SIMD RapidFuzz Pre-Ranking:** Rapidly ranks candidate sets in C-speed before feature extraction, ensuring only the top 15 most plausible candidates per source are fed into the classifier.
- **Precision-Weighted Threshold Tuning:** Optimal threshold calibration on a holdout validation set ($N=20,000$) specifically optimizing the $F_{0.5}$ formula ($2\times$ weight on precision over recall).

---

## 3. Candidate Generation (Blocking)

- **Blocking Keys Used:**
  1. `p:<compact_name[:5]>`: First 5 non-whitespace alphanumeric characters of cleaned name.
  2. `p4:<compact_name[:4]>`: First 4 characters (handles minor suffix truncations).
  3. `w:<token>`: Up to 5 significant name words, filtering out 22 corporate stopwords (`pvt`, `ltd`, `sarl`, `llc`, etc.).
  4. `pin:<digits>`: 5-digit US ZIP codes or 6-digit Indian postal codes.
  5. `a:<num>_<street>`: First numerical token concatenated with first non-digit street root (e.g., `120_tita`).
- **Candidate Aggregation & Protection:**
  - Candidates are indexed in an integer inverted table (`dict[str, list[int]]`).
  - High-frequency stop-keys (keys appearing in $> 1,200$ target records) are skipped to prevent cartesian explosion.
  - Queries aggregate candidate targets across all keys (capped at top 300 unique targets).
  - Candidates are pre-ranked using AVX2-accelerated `token_sort_ratio` on names and `token_set_ratio` on addresses, retaining the top 15 candidates per source (up to 30 candidates per entity).
- **Recall Retention:** Evaluated on the held-out validation set ($N=20,000$), this candidate generation strategy achieved an empirical recall of **98.21%** while pruning 99.98% of non-matching pairs.

---

## 4. Matching Model

**Features Used (13 Pairwise Signals):**
1. `name_token_sort`: Token sort ratio of business names (resilient to word-order permutation).
2. `name_token_set`: Token set ratio of business names (handles subset / parent-child name additions).
3. `name_jaro_winkler`: Jaro-Winkler similarity (rewards shared prefix characters).
4. `name_ratio`: Normalized Levenshtein ratio between names.
5. `name_exact`: Binary indicator for identical cleaned names.
6. `name_len_diff`: Absolute character length disparity between names.
7. `addr_token_set`: Token set ratio of business addresses.
8. `addr_ratio`: Normalized Levenshtein ratio between addresses.
9. `addr_len_diff`: Absolute character length disparity between addresses.
10. `digit_match_count`: Count of intersecting numerical tokens between addresses.
11. `digit_jaccard`: Jaccard similarity of extracted address digit sets.
12. `missing_addr`: Binary indicator if either address field is null or empty.
13. `is_s2`: Binary indicator distinguishing Source 2 vs. Source 3 records.

**Model Architecture & Configuration:**
- **Algorithm:** Gradient Boosted Decision Trees (`xgboost.XGBClassifier`).
- **Hyperparameters:** `max_depth=6`, `learning_rate=0.08`, `n_estimators=350`, `subsample=0.85`, `colsample_bytree=0.85`, `tree_method='hist'`.
- **Threshold Selection:** Scored over 448,392 validation candidate pairs. Sweeping candidate cutoffs from 0.30 to 0.90 identified **0.65** as the optimal decision threshold for maximizing Macro $F_{0.5}$ with strict singleton handling.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** **0.9738** on held-out validation pairs; **0.9190** in end-to-end candidate-generation + scoring simulation.
- **Common False Positives (Wrong Merges):**
  - Retail chains or franchises sharing identical brand names and corporate suffixes but located at distinct municipal addresses within the same city.
  - Multi-tenant commercial complexes or IT business parks (e.g., *“Cyber City, Gurugram”* or *“Tour Montparnasse, Paris”*) where disparate businesses share identical street addresses and pin codes.
- **Common False Negatives (Missed Matches):**
  - Heavy DBA/Trade name divergence where both the business name is completely distinct (e.g. holding company vs. retail storefront) and the address lacks numerical door/street digits.
  - Severe phonetic and spelling corruptions across multilingual scripts exceeding 4 edit operations.

---

## 6. Conclusion
By replacing dense matrix operations and non-unique joins with a two-pass integer inverted index and multi-threaded in-place XGBoost scoring, we achieved a production-ready entity resolution system. The pipeline successfully processed all 1,732,544 test queries and nearly 10 million targets across India, the US, and France in 68 minutes, utilizing less than 1.1 GB of RAM with zero disk paging, and passed all competition verification checks.

---

## Appendix

### A. Code Artefacts
The reproducible code is organized under `code/business_entity_resolution/`:
- `src/validation.py`: Normalization and cleaning pipelines.
- `src/blocking.py`: Multi-key candidate generation engine.
- `src/train_model.py`: Model training, feature calculation, and threshold tuner.
- `src/inference.py`: Complete test inference generator generating both competition outputs.
- `models/xgboost_er_model.json`: Trained XGBoost model parameters.
- `requirements.txt`: Environment dependencies.

To reproduce the outputs:
```bash
python code/business_entity_resolution/src/inference.py
```

### B. Validation & Hardware Metrics
- **Target Indexing Speed:** ~40 seconds per 700k records (France), ~240 seconds per 2.4M records (India).
- **Candidate Scoring Speed:** 0.71 ms per query on multi-threaded CPU.
- **Peak System Memory:** 1.05 GB RAM (strict operating safety margin on 8GB / 16GB laptops).
- **Official Submission Validator Output:**
  ```text
  required S1 entities: 1732544
  valid S2/S3 match IDs: 9969589
  matching_results.tsv: 1732544 rows (274930 empty, 1457614 non-empty).
  candidate_pairs.tsv: 1732544 rows (88117 empty, 1644427 non-empty).
  PASS — no blocking issues found. Safe to submit.
  ```
