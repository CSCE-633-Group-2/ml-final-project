import argparse
import os

import pandas as pd
from tqdm import tqdm

def process_notes_to_dataset(
    noteevents_path,
    diagnoses_icd_path,
    procedures_icd_path,
    output_path,
    chunksize=25000,
    max_rows=None,
):
    """Build row_id,text,label output from note events in a single dataframe pass."""
    notes_df = pd.read_csv(
        noteevents_path,
        compression="gzip",
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"},
        low_memory=False,
        usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "TEXT"],
    )

    diagnoses_df = pd.read_csv(
        diagnoses_icd_path,
        compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"},
        low_memory=False,
    ).drop_duplicates().reset_index(drop=True)

    procedures_df = pd.read_csv(
        procedures_icd_path,
        compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"},
        low_memory=False,
    ).drop_duplicates().reset_index(drop=True)

    notes_df["TEXT"] = notes_df["TEXT"].fillna("")

    match_cols = ["SUBJECT_ID", "HADM_ID"]
    note_index = notes_df.set_index(match_cols).index
    diagnosis_index = diagnoses_df.set_index(match_cols).index
    procedure_index = procedures_df.set_index(match_cols).index
    mask = note_index.isin(procedure_index) | note_index.isin(diagnosis_index)
    notes_df["label"] = mask.astype(int)

    notes_df_filtered_renamed = notes_df[["ROW_ID", "TEXT", "label"]].rename(
        {"ROW_ID": "row_id", "TEXT": "text"}, axis=1
    )

    if max_rows is not None:
        notes_df_filtered_renamed = notes_df_filtered_renamed.head(max_rows)

    compression = "gzip" if output_path.endswith(".gz") else None
    notes_df_filtered_renamed.to_csv(output_path, compression=compression, index=False)

    print(f"Saved {len(notes_df_filtered_renamed)} rows to {output_path}")
    label_counts = notes_df_filtered_renamed["label"].value_counts().to_dict()
    zero_count = int(label_counts.get(0, 0))
    one_count = int(label_counts.get(1, 0))
    print(f"Label counts -> 0s: {zero_count}, 1s: {one_count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build row_id,text,label dataset from MIMIC raw CSV.gz files."
    )
    parser.add_argument(
        "--noteevents-path",
        default="path/to/noteevents",
        help="Path to note events source",
    )
    parser.add_argument(
        "--output-path",
        default="path/to/output_dataset",
        help="Output CSV path for row_id,text,label",
    )
    parser.add_argument(
        "--diagnoses-icd-path",
        default="path/to/diagnoses_assignments",
        help="Path to diagnoses assignment source",
    )
    parser.add_argument(
        "--procedures-icd-path",
        default="path/to/procedures_assignments",
        help="Path to procedures assignment source",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=25000,
        help="Number of NOTE rows processed per chunk.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting output_path if it already exists.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output_path) and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_path}. "
            "Use --overwrite to replace it."
        )

    process_notes_to_dataset(
        noteevents_path=args.noteevents_path,
        diagnoses_icd_path=args.diagnoses_icd_path,
        procedures_icd_path=args.procedures_icd_path,
        output_path=args.output_path,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
