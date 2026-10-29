data_path = "C:\\Users\\KCC\\OneDrive\\Documents\\AI Wound Healing Peptide Discovery\\datasets\\cosmetic_peptide_sequences_expanded.csv"

import pandas as pd

df = pd.read_csv(data_path)

df = df.drop_duplicates(subset=["sequence_one_letter"])

with open("cosmetic_peptide_sequences.fasta", "w") as fasta_file:
    for index, row in df.iterrows():
        sequence = row["sequence_one_letter"]
        fasta_file.write(f">CP_{index}\n{sequence}\n")
