import sys
from pathlib import Path

# Add the parent folder (code/src/) to Python's import path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import polars as pl
from validation import clean_name, clean_address

# Load 10 rows from the validation file
df = pl.read_csv("D:/Coding/Mechine Learning/student_resource/dataset/val/val_source1.tsv", separator="\t").head(10)

print(f"{'RAW BUSINESS NAME':<50} | {'CLEANED NAME'}")
print("-" * 90)
for row in df.iter_rows(named=True):
    raw = row["business_name"]
    cleaned = clean_name(raw)
    print(f"{raw:<50} | {cleaned}")
