"""For a given topic dataset CSV, create three separate CSVs,
corresponding to lockdowns, masking and distancing, and vaccination

The new CSVs are written with the name of the topic suffixed to the
original filename.
"""
import argparse
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--infile", "-i", type=str, help="CSV for dataset to expand",
                    required=True)
args = parser.parse_args()

for topic in ["lockdowns", "masking and distancing", "vaccination"]:
    df = pd.read_csv(args.infile)
    cols = df.columns
    for col in cols:
        if "annotation" in col:
            df[col] = (df[col].notna() & df[col].str.contains(topic))
    new_fn = re.sub(r"\.csv$", f"_{topic.replace(' ', '_')}.csv", args.infile)
    df.to_csv(new_fn, index=False)
