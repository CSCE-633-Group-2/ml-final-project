import argparse
import os
import re

import pandas as pd
from tqdm import tqdm

def process_notes_to_dataset(
    noteevents_path,
    diagnoses_icd_path,
    procedures_icd_path,
    d_icd_diagnoses_path,
    d_icd_procedures_path,
    output_path,
    chunksize=25000,
    max_rows=None,
):
    """Build row_id,text,label output from note events in a single dataframe pass."""
    try:
        import nltk
        from nltk.corpus import stopwords as nltk_stopwords

        try:
            stopwords = set(nltk_stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stopwords = set(nltk_stopwords.words("english"))
    except Exception:
        stopwords = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is",
            "it", "its", "of", "on", "or", "that", "the", "to", "was", "were", "with", "without", "not",
            "no", "yes", "this", "these", "those", "there", "their", "his", "her", "she", "they", "them",
            "you", "your", "we", "our", "but", "if", "into", "out", "over", "under", "then", "than",
            "which", "who", "whom", "when", "where", "why", "how", "what",
        }
    min_token_len = 5

    def tokenize(text):
        return [
            token
            for token in re.findall(r"[a-z0-9]+", str(text).lower())
            if len(token) >= min_token_len and token not in stopwords
        ]

    def split_text_into_chunks(text, chunk_size=128):
        words = text.split()
        if not words:
            return [""]
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

    def build_hadm_token_map(diagnoses_df, procedures_df, d_diag_df, d_proc_df):
        hadm_tokens = {}

        def add_group_tokens(df, d_df):
            joined = df.merge(d_df, on="ICD9_CODE", how="left")
            for hadm_id, group in joined.groupby("HADM_ID"):
                tokens = set()
                for text in pd.concat([group["SHORT_TITLE"], group["LONG_TITLE"]], ignore_index=True):
                    tokens.update(tokenize(text))
                if tokens:
                    hadm_tokens.setdefault(hadm_id, set()).update(tokens)

        add_group_tokens(diagnoses_df, d_diag_df)
        add_group_tokens(procedures_df, d_proc_df)
        return hadm_tokens

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
        usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"},
        low_memory=False,
    ).drop_duplicates().reset_index(drop=True)

    procedures_df = pd.read_csv(
        procedures_icd_path,
        compression="gzip",
        usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"],
        dtype={"SUBJECT_ID": "Int64", "HADM_ID": "Int64"},
        low_memory=False,
    ).drop_duplicates().reset_index(drop=True)

    d_icd_diagnoses_df = pd.read_csv(
        d_icd_diagnoses_path,
        compression="gzip",
        usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
        low_memory=False,
    )

    d_icd_procedures_df = pd.read_csv(
        d_icd_procedures_path,
        compression="gzip",
        usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
        low_memory=False,
    )

    notes_df["TEXT"] = notes_df["TEXT"].fillna("")

    match_cols = ["SUBJECT_ID", "HADM_ID"]
    note_index = notes_df.set_index(match_cols).index
    diagnosis_index = diagnoses_df.set_index(match_cols).index
    procedure_index = procedures_df.set_index(match_cols).index
    mask = note_index.isin(procedure_index) | note_index.isin(diagnosis_index)
    notes_df["label"] = mask.astype(int)

    notes_df_filtered_renamed = notes_df[["ROW_ID", "HADM_ID", "TEXT", "label"]].rename(
        {"ROW_ID": "row_id", "TEXT": "text", "HADM_ID": "hadm_id"}, axis=1
    )

    hadm_token_map = build_hadm_token_map(
        diagnoses_df,
        procedures_df,
        d_icd_diagnoses_df,
        d_icd_procedures_df,
    )

    expanded_rows = []
    for _, row in notes_df_filtered_renamed.iterrows():
        chunks = split_text_into_chunks(row["text"], chunk_size=128)
        hadm_tokens = hadm_token_map.get(row["hadm_id"], set())
        for chunk in chunks:
            if row["label"] == 0:
                chunk_label = 0
            else:
                chunk_tokens = set(tokenize(chunk))
                chunk_label = 1 if hadm_tokens and (chunk_tokens & hadm_tokens) else 0
            expanded_rows.append(
                {
                    "row_id": row["row_id"],
                    "text": chunk,
                    "label": chunk_label,
                }
            )

    notes_df_filtered_renamed = pd.DataFrame(expanded_rows)

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
        "--d-icd-diagnoses-path",
        default="path/to/d_icd_diagnoses",
        help="Path to ICD diagnoses dictionary source",
    )
    parser.add_argument(
        "--d-icd-procedures-path",
        default="path/to/d_icd_procedures",
        help="Path to ICD procedures dictionary source",
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
        d_icd_diagnoses_path=args.d_icd_diagnoses_path,
        d_icd_procedures_path=args.d_icd_procedures_path,
        output_path=args.output_path,
        chunksize=args.chunksize,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
