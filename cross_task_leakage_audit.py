#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd


TASKS = ("Kcr", "Ksucc", "Kac")
COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("Kcr", "Ksucc"),
    ("Kcr", "Kac"),
    ("Ksucc", "Kcr"),
    ("Ksucc", "Kac"),
    ("Kac", "Kcr"),
    ("Kac", "Ksucc"),
)

# Source-data counts used in the formal pre-filter audit.
EXPECTED_COUNTS = {
    ("Kcr", "TrainReference", 1): 6975,
    ("Kcr", "TrainReference", 0): 6975,
    ("Kcr", "Test", 1): 2989,
    ("Kcr", "Test", 0): 2989,
    ("Ksucc", "TrainReference", 1): 4749,
    ("Ksucc", "TrainReference", 0): 4750,
    ("Ksucc", "Test", 1): 253,
    ("Ksucc", "Test", 0): 2973,
    ("Kac", "TrainReference", 1): 736,
    ("Kac", "TrainReference", 0): 3765,
    ("Kac", "Test", 1): 150,
    ("Kac", "Test", 0): 942,
}

FINAL_HALF_WINDOW = 15          # 31 aa total
FINAL_NEAR_IDENTITY = 0.90
FINAL_NEAR_COVERAGE = 0.80
FINAL_EXPECTED_REMOVED = 781
FINAL_ROUTE_FLAGGED = {
    ("Kcr", "Ksucc"): 193,
    ("Kcr", "Kac"): 90,
    ("Ksucc", "Kcr"): 256,
    ("Ksucc", "Kac"): 68,
    ("Kac", "Kcr"): 61,
    ("Kac", "Ksucc"): 127,
}
FINAL_REMOVAL_COUNTS = {
    ("Kcr", 1): 172,
    ("Kcr", 0): 144,
    ("Ksucc", 1): 164,
    ("Ksucc", 0): 143,
    ("Kac", 1): 33,
    ("Kac", 0): 125,
}
AA_ALLOWED = set("ARNDCQEGHILKMFPSTWYVX-")


@dataclass
class Sample:
    sample_id: str
    task: str
    split: str
    label: int
    source_file: str
    source_header: str
    protein_id: str
    site_position: Optional[int]
    sequence: str
    center_index: int
    normalized_window: str
    sequence_hash: str
    normalized_hash: str

    @property
    def site_key(self) -> str:
        if not self.protein_id or self.site_position is None:
            return ""
        return f"{self.protein_id}|{self.site_position}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KAT-Net final cross-task leakage audit (31 aa; 0.90/0.80)."
    )

    # Kcr source files
    parser.add_argument(
        "--kcr-train-pos",
        default="/home/ys/whProject/Mymodel/data/human/pos_fasta_lysine.fasta",
    )
    parser.add_argument(
        "--kcr-train-neg",
        default="/home/ys/whProject/Mymodel/data/human/neg_fasta_lysine.fasta",
    )
    parser.add_argument(
        "--kcr-test-pos",
        default="/home/ys/whProject/Mymodel/data/human/pos_ind_testdata.fasta",
    )
    parser.add_argument(
        "--kcr-test-neg",
        default="/home/ys/whProject/Mymodel/data/human/neg_ind_testdata.fasta",
    )

    # Ksucc source files
    parser.add_argument(
        "--ksucc-train-pos",
        default=(
            "/home/ys/whProject/Mymodel/data/succinylation/train/"
            "positive_sites.fasta"
        ),
    )
    parser.add_argument(
        "--ksucc-train-neg",
        default=(
            "/home/ys/whProject/Mymodel/data/succinylation/train/"
            "negative_sites.fasta"
        ),
    )
    parser.add_argument(
        "--ksucc-test-pos",
        default=(
            "/home/ys/whProject/Mymodel/data/succinylation/test/"
            "test_positive_sites.fasta"
        ),
    )
    parser.add_argument(
        "--ksucc-test-neg",
        default=(
            "/home/ys/whProject/Mymodel/data/succinylation/test/"
            "test_negative_sites.fasta"
        ),
    )

    # Kac NHAC: train + validation are both model-development data.
    parser.add_argument(
        "--kac-csv",
        default="/home/ys/whProject/Mymodel/data/acetylation/NHAC.csv",
    )
    parser.add_argument("--kac-sequence-column", default="seq_61")
    parser.add_argument(
        "--kac-development-sets",
        default="train,validation,val,valid",
    )
    parser.add_argument("--kac-test-sets", default="test")

    # Optional mapping for anonymous Kcr headers.
    parser.add_argument(
        "--kcr-mapping-csv",
        default="",
        help=(
            "Optional Kcr header mapping. Accepted columns include "
            "header/source_header, protein_id/uniprot_id and "
            "site_position/position."
        ),
    )

    # FINAL manuscript parameters. These defaults intentionally match the
    # final paper and Supplementary Table S13.
    parser.add_argument(
        "--half-window",
        type=int,
        default=FINAL_HALF_WINDOW,
        help="Residues on each side of candidate K; final setting 15 => 31 aa.",
    )
    parser.add_argument(
        "--near-identity",
        type=float,
        default=FINAL_NEAR_IDENTITY,
        help="Final near-window identity threshold; default 0.90.",
    )
    parser.add_argument(
        "--near-coverage",
        type=float,
        default=FINAL_NEAR_COVERAGE,
        help="Final near-window coverage threshold; default 0.80.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=4,
        help="Position-specific k-mer size used only for candidate indexing.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "cross_task_leakage_audit"),
    )
    parser.add_argument(
        "--skip-count-check",
        action="store_true",
        help="Skip formal source-count validation (useful for post-filter re-audit).",
    )
    parser.add_argument(
        "--expected-removed",
        type=int,
        default=FINAL_EXPECTED_REMOVED,
        help=(
            "Expected unique flagged training samples for the final pre-filter "
            "source data. Default 781. Set -1 to disable this check."
        ),
    )
    return parser.parse_args()


def normalize_sequence(value: object) -> str:
    seq = re.sub(r"\s+", "", str(value).upper())
    seq = re.sub(r"[^A-Z\-]", "X", seq)
    return "".join(aa if aa in AA_ALLOWED else "X" for aa in seq)


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_protein_id(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return ""
    if "|" in text:
        parts = [p.strip() for p in text.split("|")]
        if len(parts) >= 2 and parts[0].lower() in {"sp", "tr"}:
            text = parts[1]
    text = text.split()[0].strip(">|;,: ")
    # Canonical protein and isoform are treated as the same parent protein.
    text = re.sub(r"-\d+$", "", text)
    return text.upper()


def parse_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    header: Optional[str] = None
    parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, normalize_sequence("".join(parts))
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line)
    if header is not None:
        yield header, normalize_sequence("".join(parts))


def infer_center_index(sequence: str, preferred: Optional[int] = None) -> int:
    if not sequence:
        raise ValueError("Empty sequence.")
    if preferred is not None and 0 <= preferred < len(sequence):
        if sequence[preferred] == "K":
            return preferred
    middle = len(sequence) // 2
    if sequence[middle] == "K":
        return middle
    lysines = [i for i, aa in enumerate(sequence) if aa == "K"]
    if not lysines:
        raise ValueError(f"No lysine found: {sequence[:80]}")
    return min(lysines, key=lambda i: (abs(i - middle), i))


def center_normalize(sequence: str, center_index: int, half_window: int) -> str:
    wanted = 2 * half_window + 1
    start = center_index - half_window
    end = center_index + half_window + 1
    left_pad = max(0, -start)
    right_pad = max(0, end - len(sequence))
    core = sequence[max(0, start):min(len(sequence), end)]
    result = "-" * left_pad + core + "-" * right_pad
    if len(result) != wanted:
        raise RuntimeError(f"Normalized length {len(result)} != {wanted}")
    if result[half_window] != "K":
        raise ValueError("Candidate centre is not K after normalization.")
    return result


def find_column(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_kcr_mapping(path: str) -> Dict[str, Tuple[str, Optional[int]]]:
    if not path:
        return {}
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    hcol = find_column(df.columns, ["header", "source_header", "fasta_header", "id"])
    pcol = find_column(df.columns, ["protein_id", "uniprot_id", "uniprot", "accession"])
    scol = find_column(df.columns, ["site_position", "position", "lysine_position", "site"])
    if hcol is None or pcol is None:
        raise ValueError("Kcr mapping requires header and protein_id columns.")
    mapping: Dict[str, Tuple[str, Optional[int]]] = {}
    for _, row in df.iterrows():
        header = str(row[hcol]).strip()
        protein = canonical_protein_id(row[pcol])
        pos: Optional[int] = None
        if scol is not None and pd.notna(row[scol]):
            try:
                pos = int(float(row[scol]))
            except (TypeError, ValueError):
                pos = None
        if header:
            mapping[header] = (protein, pos)
    return mapping


def parse_ksucc_header(header: str) -> Tuple[str, Optional[int]]:
    parts = [p.strip() for p in str(header).split("|")]
    protein = canonical_protein_id(parts[1]) if len(parts) >= 2 else ""
    position: Optional[int] = None
    for token in reversed(parts):
        if re.fullmatch(r"\d+", token):
            position = int(token)
            break
    return protein, position


def parse_kac_unique_id(value: object) -> Tuple[str, Optional[int]]:
    parts = [p.strip() for p in str(value).split(";")]
    protein = canonical_protein_id(parts[0]) if parts else ""
    position: Optional[int] = None
    if len(parts) >= 2:
        try:
            position = int(float(parts[1]))
        except (TypeError, ValueError):
            position = None
    return protein, position


def make_sample(
    *, task: str, split: str, label: int, source_file: Path,
    source_header: str, protein_id: str, site_position: Optional[int],
    sequence: str, half_window: int, ordinal: int,
    preferred_center: Optional[int] = None,
) -> Sample:
    sequence = normalize_sequence(sequence)
    center = infer_center_index(sequence, preferred_center)
    norm = center_normalize(sequence, center, half_window)
    return Sample(
        sample_id=f"{task}|{split}|{source_file.name}|{ordinal}",
        task=task,
        split=split,
        label=int(label),
        source_file=str(source_file),
        source_header=str(source_header),
        protein_id=canonical_protein_id(protein_id),
        site_position=site_position,
        sequence=sequence,
        center_index=center,
        normalized_window=norm,
        sequence_hash=hash_text(sequence),
        normalized_hash=hash_text(norm),
    )


def read_fasta_samples(
    *, task: str, split: str, label: int, file_path: str,
    half_window: int, kcr_mapping: Dict[str, Tuple[str, Optional[int]]],
) -> List[Sample]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: List[Sample] = []
    for ordinal, (header, seq) in enumerate(parse_fasta(path), start=1):
        protein, position = "", None
        if task == "Ksucc":
            protein, position = parse_ksucc_header(header)
        elif task == "Kcr" and header in kcr_mapping:
            protein, position = kcr_mapping[header]
        rows.append(make_sample(
            task=task, split=split, label=label, source_file=path,
            source_header=header, protein_id=protein, site_position=position,
            sequence=seq, half_window=half_window, ordinal=ordinal,
        ))
    return rows


def read_kac_samples(
    *, file_path: str, sequence_column: str, half_window: int,
    development_sets: Set[str], test_sets: Set[str],
) -> List[Sample]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"unique_id", "label", "set", sequence_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kac CSV missing columns: {sorted(missing)}")

    rows: List[Sample] = []
    ordinal = 0
    for _, row in df.iterrows():
        set_value = str(row["set"]).strip().lower()
        if set_value in development_sets:
            split = "TrainReference"
        elif set_value in test_sets:
            split = "Test"
        else:
            continue
        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            continue
        if label not in {0, 1}:
            continue
        seq = normalize_sequence(row[sequence_column])
        if not seq:
            continue
        protein, position = parse_kac_unique_id(row["unique_id"])
        ordinal += 1
        rows.append(make_sample(
            task="Kac", split=split, label=label, source_file=path,
            source_header=str(row["unique_id"]), protein_id=protein,
            site_position=position, sequence=seq, half_window=half_window,
            ordinal=ordinal, preferred_center=len(seq) // 2,
        ))
    return rows


def samples_to_frame(samples: Sequence[Sample]) -> pd.DataFrame:
    return pd.DataFrame([asdict(s) for s in samples])


def count_summary(samples: Sequence[Sample]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for split in ("TrainReference", "Test"):
            for label in (1, 0):
                observed = sum(
                    s.task == task and s.split == split and s.label == label
                    for s in samples
                )
                expected = EXPECTED_COUNTS.get((task, split, label))
                rows.append({
                    "Task": task,
                    "Split": split,
                    "Label": label,
                    "Label_name": "Positive" if label == 1 else "Negative",
                    "Observed": observed,
                    "Expected": expected,
                    "Status": "MATCH" if expected == observed else "MISMATCH",
                })
    return pd.DataFrame(rows)


def identifier_coverage(samples: Sequence[Sample]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for split in ("TrainReference", "Test"):
            subset = [s for s in samples if s.task == task and s.split == split]
            n = len(subset)
            protein_n = sum(bool(s.protein_id) for s in subset)
            site_n = sum(bool(s.protein_id) and s.site_position is not None for s in subset)
            rows.append({
                "Task": task,
                "Split": split,
                "N": n,
                "Protein_ID_available": protein_n,
                "Protein_ID_coverage": protein_n / n if n else 0.0,
                "Site_ID_available": site_n,
                "Site_ID_coverage": site_n / n if n else 0.0,
            })
    return pd.DataFrame(rows)


def pair_row(test: Sample, train: Sample, level: str, **extra: object) -> Dict[str, object]:
    row: Dict[str, object] = {
        "Test_Task": test.task,
        "Train_Task": train.task,
        "Audit_Level": level,
        "Test_Sample_ID": test.sample_id,
        "Train_Sample_ID": train.sample_id,
        "Test_Label": test.label,
        "Train_Label": train.label,
        "Test_Protein_ID": test.protein_id,
        "Train_Protein_ID": train.protein_id,
        "Test_Site_Position": test.site_position,
        "Train_Site_Position": train.site_position,
        "Test_Source_Header": test.source_header,
        "Train_Source_Header": train.source_header,
        "Test_Normalized_Window": test.normalized_window,
        "Train_Normalized_Window": train.normalized_window,
    }
    row.update(extra)
    return row


def exact_pairs(
    test_samples: Sequence[Sample], train_samples: Sequence[Sample],
    attribute: str, audit_level: str,
) -> List[Dict[str, object]]:
    index: Dict[str, List[Sample]] = defaultdict(list)
    for sample in train_samples:
        value = getattr(sample, attribute)
        if value:
            index[str(value)].append(sample)
    rows: List[Dict[str, object]] = []
    for test in test_samples:
        value = getattr(test, attribute)
        if not value:
            continue
        for train in index.get(str(value), []):
            rows.append(pair_row(test, train, audit_level))
    return rows


def valid_residue(residue: str) -> bool:
    return residue not in {"-", "X"}


def aligned_similarity(a: str, b: str) -> Tuple[float, float, int, int]:
    if len(a) != len(b):
        raise ValueError("Normalized windows must have equal length.")
    compared = 0
    matched = 0
    for aa, bb in zip(a, b):
        if not valid_residue(aa) or not valid_residue(bb):
            continue
        compared += 1
        matched += int(aa == bb)
    identity = matched / compared if compared else 0.0
    # Final audit coverage: compared valid positions / normalized 31-aa length.
    coverage = compared / len(a) if a else 0.0
    return identity, coverage, matched, compared


def block_keys(sequence: str, block_size: int) -> List[Tuple[int, str]]:
    """Position-specific sliding k-mers used only to generate candidates."""
    keys: List[Tuple[int, str]] = []
    for start in range(0, len(sequence) - block_size + 1):
        block = sequence[start:start + block_size]
        if all(valid_residue(x) for x in block):
            keys.append((start, block))
    return keys


def near_window_pairs(
    test_samples: Sequence[Sample], train_samples: Sequence[Sample],
    identity_threshold: float, coverage_threshold: float, block_size: int,
) -> List[Dict[str, object]]:
    inverted: Dict[Tuple[int, str], Set[int]] = defaultdict(set)
    for i, train in enumerate(train_samples):
        for key in block_keys(train.normalized_window, block_size):
            inverted[key].add(i)

    rows: List[Dict[str, object]] = []
    for test in test_samples:
        candidate_indices: Set[int] = set()
        for key in block_keys(test.normalized_window, block_size):
            candidate_indices.update(inverted.get(key, set()))
        for i in sorted(candidate_indices):
            train = train_samples[i]
            # Exact normalized-window matches are reported separately.
            if test.normalized_hash == train.normalized_hash:
                continue
            identity, coverage, matched, compared = aligned_similarity(
                test.normalized_window, train.normalized_window
            )
            if identity >= identity_threshold and coverage >= coverage_threshold:
                rows.append(pair_row(
                    test, train, "Near_center_aligned_window",
                    Identity=identity,
                    Coverage=coverage,
                    Matching_positions=matched,
                    Compared_positions=compared,
                ))
    return rows


def audit_status(
    coverage: pd.DataFrame, test_task: str, train_task: str, level: str
) -> str:
    column = "Protein_ID_coverage" if level == "protein" else "Site_ID_coverage"
    t = coverage[(coverage.Task == test_task) & (coverage.Split == "Test")]
    r = coverage[(coverage.Task == train_task) & (coverage.Split == "TrainReference")]
    if t.empty or r.empty:
        return "Not_evaluable"
    a = float(t.iloc[0][column])
    b = float(r.iloc[0][column])
    if a == 1.0 and b == 1.0:
        return "Complete"
    if a == 0.0 or b == 0.0:
        return "Not_evaluable"
    return "Partial"


def subset_route(frame: pd.DataFrame, test_task: str, train_task: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[(frame.Test_Task == test_task) & (frame.Train_Task == train_task)]


def build_removal_plan(pair_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Final removal policy from processing_manifest.json:
    # exact normalized + known same protein + known same site + near windows.
    policy_levels = ("exact_normalized", "same_protein", "same_site", "near_window")
    rows: List[Dict[str, object]] = []
    for reason in policy_levels:
        frame = pair_frames[reason]
        if frame.empty:
            continue
        for _, row in frame.iterrows():
            rows.append({
                "Train_Sample_ID": str(row["Train_Sample_ID"]),
                "Train_Task": str(row["Train_Task"]),
                "Train_Label": int(row["Train_Label"]),
                "Conflicting_Test_Task": str(row["Test_Task"]),
                "Reason": reason,
                "Train_Source_Header": str(row.get("Train_Source_Header", "")),
                "Train_Protein_ID": str(row.get("Train_Protein_ID", "")),
                "Train_Site_Position": row.get("Train_Site_Position", ""),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "Train_Sample_ID", "Train_Task", "Train_Label",
            "Conflicting_Test_Tasks", "Reasons", "Train_Source_Header",
            "Train_Protein_ID", "Train_Site_Position",
        ])
    raw = pd.DataFrame(rows).drop_duplicates()

    def join_sorted(values: pd.Series) -> str:
        return ";".join(sorted({str(v) for v in values if str(v)}))

    grouped = raw.groupby(
        ["Train_Sample_ID", "Train_Task", "Train_Label"], as_index=False
    ).agg({
        "Conflicting_Test_Task": join_sorted,
        "Reason": join_sorted,
        "Train_Source_Header": "first",
        "Train_Protein_ID": "first",
        "Train_Site_Position": "first",
    }).rename(columns={
        "Conflicting_Test_Task": "Conflicting_Test_Tasks",
        "Reason": "Reasons",
    })
    return grouped.sort_values(["Train_Task", "Train_Label", "Train_Sample_ID"])


def route_summary(pair_frames: Dict[str, pd.DataFrame], coverage: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for test_task, train_task in COMPARISONS:
        flagged: Set[str] = set()
        strict: Set[str] = set()
        near: Set[str] = set()

        for level in ("exact_normalized", "same_protein", "same_site"):
            part = subset_route(pair_frames[level], test_task, train_task)
            strict.update(part.get("Train_Sample_ID", pd.Series(dtype=str)).astype(str))
        part_near = subset_route(pair_frames["near_window"], test_task, train_task)
        near.update(part_near.get("Train_Sample_ID", pd.Series(dtype=str)).astype(str))
        flagged = strict | near

        rows.append({
            "Independent_Test_Task": test_task,
            "Other_PTM_Training_Task": train_task,
            "Unique_training_samples_flagged": len(flagged),
            "Strict_conflict_training_samples": len(strict),
            "Near_window_conflict_training_samples": len(near),
            "Protein_Audit_Status": audit_status(coverage, test_task, train_task, "protein"),
            "Site_Audit_Status": audit_status(coverage, test_task, train_task, "site"),
        })
    return pd.DataFrame(rows)



def audit_all_routes(
    samples: Sequence[Sample],
    identity_threshold: float,
    coverage_threshold: float,
    block_size: int,
) -> Dict[str, pd.DataFrame]:
    """Run the six off-diagonal audits and return all pair tables."""
    all_rows: Dict[str, List[Dict[str, object]]] = {
        "same_protein": [],
        "same_site": [],
        "exact_full": [],
        "exact_normalized": [],
        "near_window": [],
    }
    for test_task, train_task in COMPARISONS:
        tests = [
            s for s in samples
            if s.task == test_task and s.split == "Test"
        ]
        trains = [
            s for s in samples
            if s.task == train_task and s.split == "TrainReference"
        ]
        all_rows["same_protein"] += exact_pairs(
            tests, trains, "protein_id", "Same_parent_protein"
        )
        all_rows["same_site"] += exact_pairs(
            tests, trains, "site_key", "Same_protein_and_lysine_site"
        )
        all_rows["exact_full"] += exact_pairs(
            tests, trains, "sequence_hash", "Exact_full_input_window"
        )
        all_rows["exact_normalized"] += exact_pairs(
            tests, trains, "normalized_hash",
            "Exact_centre_normalized_window"
        )
        all_rows["near_window"] += near_window_pairs(
            tests, trains, identity_threshold, coverage_threshold, block_size
        )

    frames: Dict[str, pd.DataFrame] = {}
    for level, rows in all_rows.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=[
                "Test_Task", "Train_Task", "Audit_Level",
                "Test_Sample_ID", "Train_Sample_ID",
                "Test_Label", "Train_Label",
            ])
        frames[level] = frame
    return frames


def build_post_filter_summary(
    pair_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize the final strict/near conflicts after removal in memory."""
    rows: List[Dict[str, object]] = []
    for test_task, train_task in COMPARISONS:
        strict_ids: Set[str] = set()
        strict_records = 0
        for level in ("exact_normalized", "same_protein", "same_site"):
            part = subset_route(pair_frames[level], test_task, train_task)
            strict_records += len(part)
            if not part.empty:
                strict_ids.update(part["Train_Sample_ID"].astype(str))
        near_part = subset_route(
            pair_frames["near_window"], test_task, train_task
        )
        near_ids = (
            set(near_part["Train_Sample_ID"].astype(str))
            if not near_part.empty else set()
        )
        rows.append({
            "Independent_Test_Task": test_task,
            "Other_PTM_Training_Task": train_task,
            "Remaining_strict_pair_records": int(strict_records),
            "Remaining_strict_training_samples": len(strict_ids),
            "Remaining_near_pair_records": int(len(near_part)),
            "Remaining_near_training_samples": len(near_ids),
        })
    return pd.DataFrame(rows)


def validate_manuscript_expected_results(
    reviewer: pd.DataFrame,
    removal_summary: pd.DataFrame,
    unique_removed: int,
) -> None:
    """Fail loudly if results drift from the final manuscript/Supplement."""
    if unique_removed != FINAL_EXPECTED_REMOVED:
        raise RuntimeError(
            f"Expected {FINAL_EXPECTED_REMOVED} unique removals, observed "
            f"{unique_removed}."
        )

    observed_routes = {
        (str(r.Independent_Test_Task), str(r.Other_PTM_Training_Task)):
            int(r.Unique_training_samples_flagged)
        for r in reviewer.itertuples(index=False)
    }
    if observed_routes != FINAL_ROUTE_FLAGGED:
        raise RuntimeError(
            "Route-level flagged counts differ from Supplementary Table S13.\n"
            f"Expected: {FINAL_ROUTE_FLAGGED}\nObserved: {observed_routes}"
        )

    observed_removal = {
        (str(r.Task), int(r.Label)): int(r.Removed_samples)
        for r in removal_summary.itertuples(index=False)
    }
    if observed_removal != FINAL_REMOVAL_COUNTS:
        raise RuntimeError(
            "Task/label removal counts differ from Supplementary Table S13.\n"
            f"Expected: {FINAL_REMOVAL_COUNTS}\nObserved: {observed_removal}"
        )

def save_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()

    # Prevent silent protocol drift.
    if args.half_window != FINAL_HALF_WINDOW:
        raise ValueError(
            f"Final manuscript uses half-window={FINAL_HALF_WINDOW} (31 aa); "
            f"received {args.half_window}."
        )
    if abs(args.near_identity - FINAL_NEAR_IDENTITY) > 1e-12:
        raise ValueError(
            f"Final manuscript uses near identity={FINAL_NEAR_IDENTITY:.2f}; "
            f"received {args.near_identity:.4f}."
        )
    if abs(args.near_coverage - FINAL_NEAR_COVERAGE) > 1e-12:
        raise ValueError(
            f"Final manuscript uses near coverage={FINAL_NEAR_COVERAGE:.2f}; "
            f"received {args.near_coverage:.4f}."
        )

    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    development_sets = {x.strip().lower() for x in args.kac_development_sets.split(",") if x.strip()}
    test_sets = {x.strip().lower() for x in args.kac_test_sets.split(",") if x.strip()}
    kcr_mapping = load_kcr_mapping(args.kcr_mapping_csv)

    input_paths = {
        "Kcr_train_positive": Path(args.kcr_train_pos).expanduser().resolve(),
        "Kcr_train_negative": Path(args.kcr_train_neg).expanduser().resolve(),
        "Kcr_test_positive": Path(args.kcr_test_pos).expanduser().resolve(),
        "Kcr_test_negative": Path(args.kcr_test_neg).expanduser().resolve(),
        "Ksucc_train_positive": Path(args.ksucc_train_pos).expanduser().resolve(),
        "Ksucc_train_negative": Path(args.ksucc_train_neg).expanduser().resolve(),
        "Ksucc_test_positive": Path(args.ksucc_test_pos).expanduser().resolve(),
        "Ksucc_test_negative": Path(args.ksucc_test_neg).expanduser().resolve(),
        "Kac_NHAC": Path(args.kac_csv).expanduser().resolve(),
    }
    for role, path in input_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{role}: {path}")

    manifest_before = pd.DataFrame([
        {
            "Role": role,
            "Path": str(path),
            "File_size_bytes": path.stat().st_size,
            "SHA256": sha256_file(path),
        }
        for role, path in input_paths.items()
    ])
    save_frame(manifest_before, out / "input_file_manifest.csv")

    print("=" * 92)
    print("KAT-Net final cross-task leakage audit")
    print("Tasks                     : Kcr / Ksucc / Kac")
    print("Same-task comparisons     : EXCLUDED")
    print("Centre-normalized window  : 31 aa")
    print("Near-window identity      : >= 0.90")
    print("Near-window coverage      : >= 0.80")
    print("Independent test files    : READ ONLY")
    print("=" * 92)

    samples: List[Sample] = []
    for label, path in ((1, args.kcr_train_pos), (0, args.kcr_train_neg)):
        samples += read_fasta_samples(
            task="Kcr", split="TrainReference", label=label, file_path=path,
            half_window=args.half_window, kcr_mapping=kcr_mapping,
        )
    for label, path in ((1, args.kcr_test_pos), (0, args.kcr_test_neg)):
        samples += read_fasta_samples(
            task="Kcr", split="Test", label=label, file_path=path,
            half_window=args.half_window, kcr_mapping=kcr_mapping,
        )
    for label, path in ((1, args.ksucc_train_pos), (0, args.ksucc_train_neg)):
        samples += read_fasta_samples(
            task="Ksucc", split="TrainReference", label=label, file_path=path,
            half_window=args.half_window, kcr_mapping={},
        )
    for label, path in ((1, args.ksucc_test_pos), (0, args.ksucc_test_neg)):
        samples += read_fasta_samples(
            task="Ksucc", split="Test", label=label, file_path=path,
            half_window=args.half_window, kcr_mapping={},
        )
    samples += read_kac_samples(
        file_path=args.kac_csv,
        sequence_column=args.kac_sequence_column,
        half_window=args.half_window,
        development_sets=development_sets,
        test_sets=test_sets,
    )

    master = samples_to_frame(samples)
    save_frame(master, out / "master_samples.csv")

    counts = count_summary(samples)
    save_frame(counts, out / "dataset_count_validation.csv")
    if not args.skip_count_check and (counts.Status != "MATCH").any():
        bad = counts[counts.Status != "MATCH"]
        raise RuntimeError(
            "Source-data counts do not match the formal audit dataset.\n"
            + bad.to_string(index=False)
        )

    coverage = identifier_coverage(samples)
    save_frame(coverage, out / "identifier_coverage.csv")

    pair_frames = audit_all_routes(
        samples,
        identity_threshold=args.near_identity,
        coverage_threshold=args.near_coverage,
        block_size=args.block_size,
    )

    filenames = {
        "same_protein": "same_protein_pairs.csv",
        "same_site": "same_site_pairs.csv",
        "exact_full": "exact_full_window_pairs.csv",
        "exact_normalized": "exact_centre_normalized_window_pairs.csv",
        "near_window": "near_window_pairs.csv",
    }
    for level, frame in pair_frames.items():
        save_frame(frame, out / filenames[level])

    reviewer = route_summary(pair_frames, coverage)
    save_frame(reviewer, out / "cross_ptm_audit_summary_for_reviewers.csv")

    removal_plan = build_removal_plan(pair_frames)
    save_frame(removal_plan, out / "removal_plan.csv")

    if removal_plan.empty:
        removal_summary = pd.DataFrame(columns=[
            "Task", "Label", "Label_name", "Removed_samples"
        ])
        unique_removed = 0
    else:
        removal_summary = (
            removal_plan.groupby(["Train_Task", "Train_Label"])
            .size().rename("Removed_samples").reset_index()
            .rename(columns={"Train_Task": "Task", "Train_Label": "Label"})
        )
        removal_summary["Label_name"] = removal_summary["Label"].map(
            {1: "Positive", 0: "Negative"}
        )
        removal_summary = removal_summary[
            ["Task", "Label", "Label_name", "Removed_samples"]
        ]
        unique_removed = int(removal_plan.Train_Sample_ID.nunique())
    save_frame(removal_summary, out / "removal_summary.csv")

    # Re-audit in memory after removing only flagged training/development samples.
    # No source or independent-test file is modified. This reproduces the
    # post-filter verification reported in Supplementary Table S13.
    removal_ids = set(removal_plan["Train_Sample_ID"].astype(str))
    filtered_samples = [
        s for s in samples
        if not (s.split == "TrainReference" and s.sample_id in removal_ids)
    ]
    filtered_master = samples_to_frame(filtered_samples)
    save_frame(filtered_master, out / "filtered_master_samples_in_memory.csv")
    post_pair_frames = audit_all_routes(
        filtered_samples,
        identity_threshold=args.near_identity,
        coverage_threshold=args.near_coverage,
        block_size=args.block_size,
    )
    post_filter = build_post_filter_summary(post_pair_frames)
    save_frame(post_filter, out / "post_filter_audit.csv")
    post_filter_pass = bool(
        (post_filter["Remaining_strict_pair_records"] == 0).all()
        and (post_filter["Remaining_near_pair_records"] == 0).all()
    )

    # Verify audit script itself did not alter any original/test input file.
    manifest_after = pd.DataFrame([
        {
            "Role": role,
            "Path": str(path),
            "File_size_bytes": path.stat().st_size,
            "SHA256": sha256_file(path),
        }
        for role, path in input_paths.items()
    ])
    unchanged = bool(
        (manifest_before["SHA256"].values == manifest_after["SHA256"].values).all()
        and (manifest_before["File_size_bytes"].values == manifest_after["File_size_bytes"].values).all()
    )

    kcr_rows = coverage[coverage.Task == "Kcr"]
    kcr_limited = bool(
        (kcr_rows.Protein_ID_coverage < 1.0).any()
        or (kcr_rows.Site_ID_coverage < 1.0).any()
    )

    config = {
        "tasks": list(TASKS),
        "comparisons": [
            {"test_task": a, "train_task": b} for a, b in COMPARISONS
        ],
        "same_task_comparisons_excluded": True,
        "center_normalized_window_length": 31,
        "half_window": args.half_window,
        "near_identity_threshold": args.near_identity,
        "near_coverage_threshold": args.near_coverage,
        "candidate_index_block_size": args.block_size,
        "removal_policy": [
            "exact_centre_normalized_window",
            "known_same_parent_protein",
            "known_same_protein_and_lysine_site",
            "near_center_aligned_window",
        ],
        "kac_train_reference": "train + validation",
        "independent_test_files_modified": False,
        "kcr_identifier_limitation": kcr_limited,
        "expected_unique_removed_training_samples": FINAL_EXPECTED_REMOVED,
        "expected_route_flagged_counts": {
            f"{a}_test_vs_{b}_train": n
            for (a, b), n in FINAL_ROUTE_FLAGGED.items()
        },
        "expected_task_label_removal_counts": {
            f"{task}|{label}": n
            for (task, label), n in FINAL_REMOVAL_COUNTS.items()
        },
    }
    (out / "audit_configuration.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = {
        "status": (
            "AUDIT_COMPLETE_WITH_KCR_IDENTIFIER_LIMITATION"
            if kcr_limited else "AUDIT_COMPLETE"
        ),
        "unique_flagged_training_samples": unique_removed,
        "expected_unique_flagged_training_samples": args.expected_removed,
        "input_files_unchanged": unchanged,
        "independent_test_files_modified": False,
        "near_identity_threshold": args.near_identity,
        "near_coverage_threshold": args.near_coverage,
        "window_length": 31,
        "post_filter_zero_conflicts": post_filter_pass,
    }
    (out / "audit_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.skip_count_check:
        validate_manuscript_expected_results(
            reviewer, removal_summary, unique_removed
        )
    if not post_filter_pass:
        raise RuntimeError(
            "Post-filter re-audit failed: strict or near-window conflicts remain."
        )

    print("\nCross-task route summary:")
    print(reviewer.to_string(index=False))
    print("\nRemoval summary:")
    print(removal_summary.to_string(index=False))
    print(f"\nUnique flagged training samples: {unique_removed}")
    print(f"Input/test files unchanged      : {unchanged}")
    print(f"Post-filter zero conflicts      : {post_filter_pass}")
    if kcr_limited:
        print(
            "Kcr identifier limitation       : YES; Kcr-related protein/site "
            "independence is not claimed."
        )

    if not unchanged:
        raise RuntimeError("An input file changed during the audit.")

    if args.expected_removed >= 0 and not args.skip_count_check:
        if unique_removed != args.expected_removed:
            raise RuntimeError(
                f"Expected {args.expected_removed} unique flagged training samples "
                f"for the final pre-filter audit, but observed {unique_removed}. "
                "Check source files, file versions and paths."
            )

    print(f"\n[PASS] Audit completed. Outputs: {out}")
    print(
        "For the final manuscript workflow, the removal plan is subsequently "
        "applied only to training/development samples; independent tests remain "
        "unchanged, followed by a post-filter re-audit."
    )


if __name__ == "__main__":
    main()