from __future__ import annotations

import io
import itertools
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common.model_sync import sync_model_weights, weights_dir_for

from .property_tables import PROPERTY_TABLES

CODE_DIR = Path(__file__).resolve().parent
MODEL_DIR = weights_dir_for(CODE_DIR)
MODEL_PATH = MODEL_DIR / "model.joblib"

AMINO_ACID_SET = set("ACDEFGHIKLMNPQRSTVWY")


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty.")
    if any(aa not in AMINO_ACID_SET for aa in sequence):
        raise ValueError(
            f"sequence contains non-standard amino acids: {sorted(set(sequence) - AMINO_ACID_SET)}"
        )
    if not (6 <= len(sequence) <= 50):
        raise ValueError(
            f"sequence length {len(sequence)} outside the model's trained range [6, 50]"
        )


class Extractor:
    """In-process reimplementation of the author's composition_calculate_hemopi2_2.py.

    Each ``_feature_*`` method reproduces one descriptor group's exact arithmetic
    from the original script, but operates on in-memory sequences/DataFrames
    instead of round-tripping through stdout-redirected CSV files.
    """

    STD = "ACDEFGHIKLMNPQRSTVWY"

    MOLECULAR_WEIGHT_TABLE = {
        "A": 0.089,
        "R": 0.174,
        "N": 0.132,
        "D": 0.133,
        "C": 0.121,
        "E": 0.147,
        "Q": 0.146,
        "G": 0.075,
        "H": 0.155,
        "I": 0.131,
        "L": 0.131,
        "K": 0.146,
        "M": 0.150,
        "F": 0.165,
        "P": 0.115,
        "S": 0.105,
        "T": 0.119,
        "W": 0.204,
        "Y": 0.181,
        "V": 0.117,
    }
    WATER_WEIGHT = 0.018

    PCP_HEADERS = [
        "PCP_PC",
        "PCP_NC",
        "PCP_NE",
        "PCP_PO",
        "PCP_NP",
        "PCP_AL",
        "PCP_CY",
        "PCP_AR",
        "PCP_AC",
        "PCP_BS",
        "PCP_NE_pH",
        "PCP_HB",
        "PCP_HL",
        "PCP_NT",
        "PCP_HX",
        "PCP_SC",
        "PCP_SS_HE",
        "PCP_SS_ST",
        "PCP_SS_CO",
        "PCP_SA_BU",
        "PCP_SA_EX",
        "PCP_SA_IN",
        "PCP_TN",
        "PCP_SM",
        "PCP_LR",
        "PCP_Z1",
        "PCP_Z2",
        "PCP_Z3",
        "PCP_Z4",
        "PCP_Z5",
    ]
    PRI_HEADERS = [
        "PRI_PC",
        "PRI_NC",
        "PRI_NE",
        "PRI_PO",
        "PRI_NP",
        "PRI_AL",
        "PRI_CY",
        "PRI_AR",
        "PRI_AC",
        "PRI_BS",
        "PRI_NE_pH",
        "PRI_HB",
        "PRI_HL",
        "PRI_NT",
        "PRI_HX",
        "PRI_SC",
        "PRI_SS_HE",
        "PRI_SS_ST",
        "PRI_SS_CO",
        "PRI_SA_BU",
        "PRI_SA_EX",
        "PRI_SA_IN",
        "PRI_TN",
        "PRI_SM",
        "PRI_LR",
    ]
    AMINO_ACID_INDEX = {
        aa: i
        for i, aa in enumerate(
            [
                "A",
                "C",
                "D",
                "E",
                "F",
                "G",
                "H",
                "I",
                "K",
                "L",
                "M",
                "N",
                "P",
                "Q",
                "R",
                "S",
                "T",
                "V",
                "W",
                "Y",
            ]
        )
    }
    CTC_GROUP = {
        "A": "1",
        "G": "1",
        "V": "1",
        "I": "2",
        "L": "2",
        "F": "2",
        "P": "2",
        "Y": "3",
        "M": "3",
        "T": "3",
        "S": "3",
        "H": "4",
        "N": "4",
        "Q": "4",
        "W": "4",
        "R": "5",
        "K": "5",
        "D": "6",
        "E": "6",
        "C": "7",
    }
    CTC_TRIADS = ["".join(t) for t in itertools.product("1234567", repeat=3)]

    def __init__(self, property_tables):
        self.atom = self._read_csv_text(property_tables["atom.csv"], header=None)
        self.bonds = pd.read_csv(io.StringIO(property_tables["bonds.csv"]), sep=",")
        self.pcp = self._read_csv_text(
            property_tables["PhysicoChemical.csv"], header=None
        )
        self.attr = pd.read_csv(
            io.StringIO(property_tables["aa_attr_group.csv"]), sep="\t"
        )
        self.paac_data = pd.read_csv(io.StringIO(property_tables["data"]), sep="\t")
        self.schneider_wrede = pd.read_csv(
            io.StringIO(property_tables["Schneider-Wrede.csv"]), index_col="Name"
        )
        self.grantham = pd.read_csv(
            io.StringIO(property_tables["Grantham.csv"]), index_col="Name"
        )

    @staticmethod
    def _read_csv_text(text, **kwargs):
        return pd.read_csv(io.StringIO(text), **kwargs)

    def extract(self, sequences):
        sequences = [s.upper() for s in sequences]
        groups = [
            self._feature_molecular_weight(sequences),
            self._feature_length(sequences),
            self._feature_aac(sequences),
            self._feature_dpc(sequences, q=1),
            self._feature_atc(sequences),
            self._feature_btc(sequences),
            self._feature_pcp(sequences),
            self._feature_rri(sequences),
            self._feature_pri(sequences),
            self._feature_ddr(sequences),
            self._feature_ser(sequences),
            self._feature_sep(sequences),
            self._feature_ctc(sequences),
            self._feature_cetd(sequences),
            self._feature_paac(sequences, lambdaval=1),
            self._feature_apaac(sequences, lambdaval=1),
            self._feature_qso(sequences, gap=1),
            self._feature_soc(sequences, gap=1),
        ]
        return pd.concat(groups, axis=1)

    def _feature_molecular_weight(self, sequences):
        values = [
            sum(self.MOLECULAR_WEIGHT_TABLE.get(aa, 0) for aa in s)
            - self.WATER_WEIGHT * (len(s) - 1)
            for s in sequences
        ]
        return pd.DataFrame({"Molecular Weight (kDa)": [round(v, 3) for v in values]})

    def _feature_length(self, sequences):
        return pd.DataFrame({"length": [len(s) for s in sequences]})

    def _feature_aac(self, sequences):
        rows = []
        for s in sequences:
            rows.append([round((s.count(i) / len(s)) * 100, 2) for i in self.STD])
        return pd.DataFrame(rows, columns=[f"AAC_{i}" for i in self.STD])

    def _feature_dpc(self, sequences, q):
        columns = [f"DPC{q}_{a}{b}" for a in self.STD for b in self.STD]
        rows = []
        for s in sequences:
            row = []
            denom = len(s) - q
            for a in self.STD:
                for b in self.STD:
                    pair = a + b
                    count = sum(
                        1
                        for m3 in range(0, len(s) - q)
                        if s[m3 : m3 + q + 1 : q] == pair
                    )
                    row.append(round((count / denom) * 100, 2))
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _feature_atc(self, sequences):
        atom = self.atom.copy()
        atom["C_atom"] = atom[1].apply(lambda v: v.count("C"))
        atom["H_atom"] = atom[1].apply(lambda v: v.count("H"))
        atom["N_atom"] = atom[1].apply(lambda v: v.count("N"))
        atom["O_atom"] = atom[1].apply(lambda v: v.count("O"))
        atom["S_atom"] = atom[1].apply(lambda v: v.count("S"))
        atom_index = {
            row[0].replace(" ", ""): row for row in atom.itertuples(index=False)
        }
        rows = []
        for s in sequences:
            totals = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0}
            for ch in s:
                row = atom_index.get(ch)
                if row is not None:
                    # The author script builds these DataFrame columns in the order
                    # C_atom, O_atom, H_atom, N_atom, S_atom but then reads them back
                    # by fixed position as if the order were C, H, N, O, S. That
                    # off-by-one column shift is preserved here since the released
                    # model was trained on its (mislabeled) output.
                    totals["C"] += row.C_atom
                    totals["H"] += row.O_atom
                    totals["N"] += row.H_atom
                    totals["O"] += row.N_atom
                    totals["S"] += row.S_atom
            count = sum(totals.values())
            rows.append(
                [round((totals[k] / count) * 100, 2) for k in ("C", "H", "N", "O", "S")]
            )
        return pd.DataFrame(rows, columns=["ATC_C", "ATC_H", "ATC_N", "ATC_O", "ATC_S"])

    def _feature_btc(self, sequences):
        bonds = self.bonds
        bond_index = {row[0]: row for row in bonds.itertuples(index=False)}
        rows = []
        for s in sequences:
            tot = h = si = du = 0
            for ch in s:
                row = bond_index.get(ch)
                if row is not None:
                    tot += row.nBonds_tot
                    h += row.Hydrogen_bonds
                    si += row.nBondsS
                    du += row.nBondsD
            rows.append([tot, h, si, du])
        return pd.DataFrame(rows, columns=["BTC_T", "BTC_H", "BTC_S", "BTC_D"])

    def _pcp_lookup(self, peptide, feature_num):
        total = 0.0
        for aa in peptide:
            idx = self.AMINO_ACID_INDEX.get(aa)
            total += self.pcp[idx][feature_num]
        return total

    def _feature_pcp(self, sequences):
        n_features = self.pcp.shape[0]
        rows = []
        for s in sequences:
            row = []
            for j in range(n_features):
                val = self._pcp_lookup(s, j)
                row.append(round(val / len(s), 3) if len(s) != 0 else float("nan"))
            rows.append(row)
        return pd.DataFrame(rows, columns=self.PCP_HEADERS)

    def _feature_rri(self, sequences):
        # The author script's `count` and `x` counters are never reset across amino
        # acids or sequences (only re-initialized once, before the outermost loop),
        # so their leftover values leak into the next iteration. That stateful bug
        # is preserved here since the released model was trained on its output.
        columns = [f"RRI_{i}" for i in self.STD]
        rows = []
        count = 0
        x = 0
        for s in sequences:
            row = []
            for aa in self.STD:
                cc = []
                for ch in s:
                    if ch == aa:
                        count += 1
                        cc.append(count)
                    else:
                        count = 0
                while x < len(cc):
                    if x + 1 < len(cc):
                        if cc[x] != cc[x + 1]:
                            if cc[x] < cc[x + 1]:
                                cc[x] = 0
                    x += 1
                cc1 = [e for e in cc if e != 0]
                cc_sq = [e * e for e in cc if e != 0]
                zz = sum(cc_sq)
                zz1 = sum(cc1)
                zz2 = zz / zz1 if zz1 != 0 else 0
                row.append(round(zz2, 2))
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _pri_lookup(self, peptide, feature_num):
        return [self.pcp[self.AMINO_ACID_INDEX[aa]][feature_num] for aa in peptide]

    def _feature_pri(self, sequences):
        rows = []
        for s in sequences:
            row = []
            for j in range(25):
                bin_prof = self._pri_lookup(s, j)
                k = num = ones = 0
                for idx, val in enumerate(bin_prof):
                    if val == 0:
                        num += k * k
                        k = 0
                    else:
                        k += 1
                        ones += 1
                    if idx == len(bin_prof) - 1 and val != 0:
                        num += k * k
                row.append(round(num / (ones * ones), 2) if ones != 0 else 0)
            rows.append(row)
        return pd.DataFrame(rows, columns=self.PRI_HEADERS)

    def _feature_ddr(self, sequences):
        columns = [f"DDR_{i}" for i in self.STD]
        rows = []
        for s in sequences:
            rev = s[::-1]
            row = []
            for aa in self.STD:
                zz = [pos for pos, ch in enumerate(s) if ch == aa]
                pp = [pos for pos, ch in enumerate(rev) if ch == aa]
                ss = [zz[i + 1] - zz[i] - 1 for i in range(len(zz) - 1)]
                if zz:
                    ss.insert(0, zz[0])
                    ss.insert(len(ss), pp[0])
                cc1 = sum(ss) + 1
                cc = sum(e * e for e in ss)
                row.append(round(cc / cc1, 2))
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _feature_ser(self, sequences):
        columns = [f"SER_{i}" for i in self.STD]
        rows = []
        for s in sequences:
            counts = Counter(s)
            length = len(s)
            row = []
            for aa in self.STD:
                freq = counts.get(aa, 0)
                row.append(
                    round((freq / length) * math.log(freq / length, 2), 3)
                    if freq
                    else 0.0
                )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    def _feature_sep(self, sequences):
        values = []
        for s in sequences:
            counts, length = Counter(s), len(s)
            values.append(
                round(
                    -sum(f / length * math.log(f / length, 2) for f in counts.values()),
                    3,
                )
            )
        return pd.DataFrame({"SEP": values})

    def _feature_ctc(self, sequences):
        columns = [f"CTC_{t}" for t in self.CTC_TRIADS]
        rows = []
        for s in sequences:
            coded = "".join(self.CTC_GROUP[ch] for ch in s)
            occ = []
            for triad in self.CTC_TRIADS:
                count = 0
                beg = 0
                while True:
                    beg = coded.find(triad, beg)
                    if beg == -1:
                        break
                    count += 1
                    beg += 1
                occ.append(count)
            min_occ, max_occ = min(occ), max(occ)
            rows.append([round((o - min_occ) / max_occ, 3) for o in occ])
        return pd.DataFrame(rows, columns=columns)

    def _feature_cetd(self, sequences):
        attr = self.attr
        # The author script matches residues by scanning each raw category string
        # character-by-character (not splitting on commas), so build the lookup the
        # same way: some rows have stray formatting (e.g. a space instead of a comma)
        # that only ever collides with non-residue characters and is otherwise inert.
        group_of = []  # group_of[attribute_index]: residue char -> group code (1,2,3)
        for i in range(len(attr)):
            groups_by_cat = {}
            for cat in (1, 2, 3):
                for ch in str(attr.iloc[i, cat]):
                    groups_by_cat[ch] = cat
            group_of.append(groups_by_cat)

        header1 = [
            "CeTD_HB",
            "CeTD_VW",
            "CeTD_PO",
            "CeTD_PZ",
            "CeTD_CH",
            "CeTD_SS",
            "CeTD_SA",
        ]
        header2 = [
            "CeTD_11",
            "CeTD_12",
            "CeTD_13",
            "CeTD_21",
            "CeTD_22",
            "CeTD_23",
            "CeTD_31",
            "CeTD_32",
            "CeTD_33",
        ]
        header4 = ["HB", "VW", "PO", "PZ", "CH", "SS", "SA"]
        header3 = ["CeTD_0_p", "CeTD_25_p", "CeTD_50_p", "CeTD_75_p", "CeTD_100_p"]

        comp_rows, trans_rows, dist_rows = [], [], []
        for s in sequences:
            # One category table row is missing a residue in its source data
            # (e.g. 'C' absent from every group of "normalized vander Waals
            # volume"); the author script's matching loop simply finds no
            # match and appends nothing for that residue at that row, so the
            # per-attribute code list can be shorter than the sequence.
            per_attr_codes = [
                [group_of[i][ch] for ch in s if ch in group_of[i]]
                for i in range(len(attr))
            ]

            comp_row = []
            for codes in per_attr_codes:
                n = len(codes)
                comp_row.extend(round((codes.count(k) / n) * 100, 2) for k in (1, 2, 3))
            comp_rows.append(comp_row)

            trans_row = []
            for codes in per_attr_codes:
                pairs = list(zip(codes, codes[1:]))
                for a, b in [
                    (1, 1),
                    (1, 2),
                    (1, 3),
                    (2, 1),
                    (2, 2),
                    (2, 3),
                    (3, 1),
                    (3, 2),
                    (3, 3),
                ]:
                    trans_row.append(sum(1 for p in pairs if p == (a, b)))
            trans_rows.append(trans_row)

            dist_row = []
            for codes in per_attr_codes:
                for k in (1, 2, 3):
                    positions = [idx for idx, v in enumerate(codes) if v == k]
                    n = len(positions)
                    for pct in (0, 25, 50, 75, 100):
                        dist_row.append(math.floor((pct * n) / 100))
            dist_rows.append(dist_row)

        comp_columns = [f"{h}{n}" for h in header1 for n in (1, 2, 3)]
        comp_df = pd.DataFrame(comp_rows, columns=comp_columns)

        trans_columns = [f"{h2}_{h4}" for h2 in header2 for h4 in header4]
        trans_df = pd.DataFrame(trans_rows, columns=trans_columns)

        dist_columns = [
            f"{h3}_{h4}{n}" for n in (1, 2, 3) for h4 in header4 for h3 in header3
        ]
        dist_df = pd.DataFrame(dist_rows, columns=dist_columns)

        return pd.concat([comp_df, trans_df, dist_df], axis=1)

    def _standardize(self, rows):
        """Column-standardize a 3xN property matrix using population std, matching the author script."""
        standardized = []
        for row in rows:
            mean = sum(row) / len(row)
            rr = math.sqrt(sum((p - mean) ** 2 for p in row) / len(row))
            standardized.append([(p - mean) / rr for p in row])
        return standardized

    def _feature_paac(self, sequences, lambdaval, w=0.05):
        data1 = self.paac_data
        aa_index = {aa: i for i, aa in enumerate(self.STD)}
        dd = self._standardize([list(data1.iloc[i][1:]) for i in range(3)])

        head = [f"PAAC{lambdaval}_lam{n}" for n in range(1, lambdaval + 1)]
        rows = []
        for s in sequences:
            cc = []
            for n in range(1, lambdaval + 1):
                terms = []
                for p in range(len(s) - n):
                    a, b = s[p], s[p + n]
                    diff = sum(
                        (dd[i][aa_index[a]] - dd[i][aa_index[b]]) ** 2
                        for i in range(len(dd))
                    ) / len(dd)
                    terms.append(diff)
                cc.append(sum(terms) / (len(s) - n))
            denom = 1 + w * sum(cc)
            rows.append([(w * p) / denom for p in cc])
        lam_df = pd.DataFrame(rows, columns=head).round(4)

        aac_columns = [f"PAAC{lambdaval}_{i}" for i in self.STD]
        aac_df = self._feature_aac(sequences)
        aac_df.columns = aac_columns
        return pd.concat(
            [aac_df.reset_index(drop=True), lam_df.reset_index(drop=True)], axis=1
        )

    def _feature_apaac(self, sequences, lambdaval, w=0.05):
        data1 = self.paac_data
        aa_index = {aa: i for i, aa in enumerate(self.STD)}
        dd = self._standardize([list(data1.iloc[i][1:]) for i in range(3)])

        head = [
            f"APAAC{lambdaval}_{e}_lam{n}"
            for n in range(1, lambdaval + 1)
            for e in ("HB", "HL", "SC")
        ]
        rows = []
        for s in sequences:
            cc = []
            for n in range(1, lambdaval + 1):
                for b in range(len(dd)):
                    terms = [
                        dd[b][aa_index[s[p]]] * dd[b][aa_index[s[p + n]]]
                        for p in range(len(s) - n)
                    ]
                    cc.append(sum(terms) / (len(s) - n))
            denom = 1 + w * sum(cc)
            rows.append([(w * p) / denom for p in cc])
        lam_df = pd.DataFrame(rows, columns=head).round(4)

        aac_columns = [f"APAAC{lambdaval}_{i}" for i in self.STD]
        aac_df = self._feature_aac(sequences)
        aac_df.columns = aac_columns
        return pd.concat(
            [aac_df.reset_index(drop=True), lam_df.reset_index(drop=True)], axis=1
        )

    def _feature_qso(self, sequences, gap, w=0.1):
        mat1, mat2 = self.schneider_wrede, self.grantham
        h1 = [f"QSO{gap}_SC_{aa}" for aa in self.STD]
        h2 = [f"QSO{gap}_G_{aa}" for aa in self.STD]
        h3 = [f"QSO{gap}_SC{n}" for n in range(1, gap + 1)]
        h4 = [f"QSO{gap}_G{n}" for n in range(1, gap + 1)]

        rows = []
        for s in sequences:
            sc_terms, g_terms = [], []
            for n in range(1, gap + 1):
                sc_terms.append(
                    sum(mat1[s[j]][s[j + n]] ** 2 for j in range(len(s) - n))
                )
                g_terms.append(
                    sum(mat2[s[j]][s[j + n]] ** 2 for j in range(len(s) - n))
                )
            sc_sum, g_sum = sum(sc_terms), sum(g_terms)
            row = []
            for aa in self.STD:
                count = s.count(aa)
                row.append(round(count / (1 + w * sc_sum), 4))
            for aa in self.STD:
                count = s.count(aa)
                row.append(round(count / (1 + w * g_sum), 4))
            for term in sc_terms:
                row.append(round((w * term) / (1 + w * sc_sum), 4))
            for term in sc_terms:
                row.append(round((w * term) / (1 + w * sc_sum), 4))
            rows.append(row)
        return pd.DataFrame(rows, columns=h1 + h2 + h3 + h4)

    def _feature_soc(self, sequences, gap):
        mat1, mat2 = self.schneider_wrede, self.grantham
        h1 = [f"SOC{gap}_SC{n}" for n in range(1, gap + 1)]
        h2 = [f"SOC{gap}_G{n}" for n in range(1, gap + 1)]

        rows = []
        for s in sequences:
            row = []
            for n in range(1, gap + 1):
                sc_sum = sum(mat1[s[j]][s[j + n]] ** 2 for j in range(len(s) - n)) / (
                    len(s) - n
                )
                g_running = 0.0
                for j in range(len(s) - n):
                    g_running += mat2[s[j]][s[j + n]] ** 2
                g_sum = g_running / (len(s) - n)
                row.append(sc_sum)
                row.append(g_sum)
            sc_vals = row[0::2]
            g_vals = row[1::2]
            rows.append(sc_vals + g_vals)
        return pd.DataFrame(rows, columns=h1 + h2).round(4)


def normalize_sequences(values):
    values = pd.Series(values, dtype="string")
    sequences = values.str.strip()
    valid = (
        sequences.notna()
        & sequences.str.fullmatch("[ACDEFGHIKLMNPQRSTVWY]+")
        & sequences.str.len().between(6, 50)
    ).fillna(False)
    if not valid.all():
        raise ValueError(
            f"Invalid sequences at zero-based rows {list(values.index[~valid])}"
        )
    return sequences


def _y_to_phc50(y_pred: np.ndarray) -> np.ndarray:
    """Model output is -log10(HC50 in uM). Convert to this codebase's
    pHC50 = -log10(HC50 in M) convention: HC50_M = HC50_uM * 1e-6, so
    -log10(HC50_M) = -log10(HC50_uM) + 6 = y_pred + 6. Same convention and
    offset as hemolysis_predictor_v2's HC50(uM) -> pHC50 conversion."""
    return y_pred + 6.0


class ReplicatedHemoPI2Predictor:
    """Lazy-loaded sklearn Pipeline (imputer + RandomForestRegressor),
    trained offline from HemoPI2's own published cross-validation/test
    splits. Returns pHC50 in this codebase's convention (see _y_to_phc50)."""

    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = Path(model_path)
        self.loaded = False

    def _load(self) -> None:
        if self.loaded:
            return
        sync_model_weights(CODE_DIR)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {self.model_path}")
        self.model = joblib.load(self.model_path)
        self.loaded = True

    def _extract_cached(self, sequences: list[str], feature_extractor) -> pd.DataFrame:
        normalized = list(normalize_sequences(sequences))
        rows = feature_extractor.get_hemolysis_v1_descriptors_batch(normalized)
        return pd.DataFrame(rows)

    def predict_phc50(self, sequence: str, feature_extractor: "FeatureExtractor") -> float:
        _validate_sequence(sequence)
        self._load()
        X = self._extract_cached([sequence], feature_extractor)
        y_pred = self.model.predict(X)
        return float(_y_to_phc50(y_pred)[0])

    def predict_phc50_batch(
        self,
        sequences: list[str],
        feature_extractor: "FeatureExtractor",
    ) -> list[float]:
        """Batched predict_phc50: one descriptor-extraction pass and one
        model.predict() call for every sequence, instead of one call per
        sequence -- same batching benefit as hemolysis_predictor_v2, without
        the subprocess overhead."""
        for i, sequence in enumerate(sequences):
            try:
                _validate_sequence(sequence)
            except ValueError as exc:
                raise ValueError(f"sequence at index {i} invalid: {exc}") from exc
        if not sequences:
            return []
        self._load()
        X = self._extract_cached(sequences, feature_extractor)
        y_pred = self.model.predict(X)
        return [float(v) for v in _y_to_phc50(y_pred)]
