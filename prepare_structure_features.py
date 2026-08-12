#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prepare AlphaFold2/ColabFold structural features for KAT-Net."""

import argparse
import gzip
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import DSSP, PDBParser
from Bio.PDB.Polypeptide import is_aa


VALID_AA = set("ACDEFGHIKLMNPQRSTVWYX-")
FEATURE_KEYS = ("coords", "plddt", "sasa", "ss", "pae", "disto")


def safe_core_key(value):
    value = str(value)
    if not value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid Core_Key: {value}")
    return value


def clean_sequence(sequence):
    return re.sub(r"\s+", "", str(sequence).upper())


def write_fasta(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for name, sequence in records:
            f.write(f">{name}\n{sequence}\n")


# Prepare unique structural samples.
def prepare_manifest(args):
    df = pd.read_csv(args.csv_path)

    required = {"Core_Key", "Dynamic_Sequence", "Center_K_Index"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    rows = []

    for _, row in df.iterrows():
        key = safe_core_key(row["Core_Key"])
        sequence = clean_sequence(row["Dynamic_Sequence"])
        center = int(row["Center_K_Index"])

        if not sequence:
            raise ValueError(f"Empty sequence: {key}")

        invalid = set(sequence) - VALID_AA
        if invalid:
            raise ValueError(f"{key}: invalid residues {sorted(invalid)}")

        if args.window_length is not None and len(sequence) != args.window_length:
            raise ValueError(
                f"{key}: length={len(sequence)}, expected={args.window_length}"
            )

        if center < 0 or center >= len(sequence) or sequence[center] != "K":
            raise ValueError(f"{key}: invalid center K")

        if args.center_index is not None and center != args.center_index:
            raise ValueError(
                f"{key}: Center_K_Index={center}, expected={args.center_index}"
            )

        model_sequence = sequence.strip("-")

        if "-" in model_sequence:
            raise ValueError(f"{key}: internal '-' is not allowed")

        if not model_sequence:
            raise ValueError(f"{key}: sequence contains only padding")

        real_positions = [i for i, aa in enumerate(sequence) if aa != "-"]

        if center not in real_positions:
            raise ValueError(f"{key}: center K is padding")

        rows.append({
            "Core_Key": key,
            "Dynamic_Sequence": sequence,
            "Center_K_Index": center,
            "Model_Sequence": model_sequence,
            "Center_Model_Index": real_positions.index(center),
            "Real_Positions": ",".join(map(str, real_positions)),
        })

    manifest = pd.DataFrame(rows)

    # One Core_Key must represent one structural sequence.
    conflict = (
        manifest.groupby("Core_Key")[["Dynamic_Sequence", "Center_K_Index"]]
        .nunique()
        .max(axis=1)
        .gt(1)
    )

    if conflict.any():
        bad = conflict[conflict].index.tolist()[:10]
        raise ValueError(f"Conflicting Core_Key records: {bad}")

    manifest = manifest.drop_duplicates("Core_Key").reset_index(drop=True)
    manifest.insert(0, "AF2_ID", [f"kat_{i:08d}" for i in range(len(manifest))])

    work_dir = args.work_dir
    fasta_dir = work_dir / "fasta"

    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)

    manifest.to_csv(work_dir / "structure_manifest.csv", index=False)

    for path in fasta_dir.glob("chunk_*.fasta"):
        path.unlink()

    for chunk_id, start in enumerate(
        range(0, len(manifest), args.chunk_size), start=1
    ):
        part = manifest.iloc[start:start + args.chunk_size]
        records = zip(part["AF2_ID"], part["Model_Sequence"])

        write_fasta(
            records,
            fasta_dir / f"chunk_{chunk_id:04d}.fasta",
        )

    print(f"Samples       : {len(manifest):,}")
    print(f"FASTA chunks  : {len(list(fasta_dir.glob('chunk_*.fasta'))):,}")
    print(f"Manifest      : {work_dir / 'structure_manifest.csv'}")


# Run the same ColabFold configuration used in the external Kcr workflow.
def run_colabfold(args):
    executable = shutil.which(args.colabfold_bin)

    if executable is None:
        candidate = Path(args.colabfold_bin)
        if candidate.is_file():
            executable = str(candidate.resolve())

    if executable is None:
        raise FileNotFoundError("colabfold_batch was not found")

    fasta_dir = args.work_dir / "fasta"
    output_root = args.work_dir / "colabfold_output"

    if not fasta_dir.is_dir():
        raise FileNotFoundError("Run --stage prepare first")

    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for fasta in sorted(fasta_dir.glob("chunk_*.fasta")):
        output_dir = output_root / fasta.stem

        if output_dir.exists():
            if args.overwrite:
                shutil.rmtree(output_dir)
            else:
                print(f"Skip existing: {fasta.name}")
                continue

        output_dir.mkdir(parents=True)

        command = [executable]

        if args.colabfold_data is not None:
            command += ["--data", str(args.colabfold_data.resolve())]

        command += [
            "--msa-mode", "single_sequence",
            "--num-models", "1",
            "--model-type", "alphafold2_ptm",
            "--num-recycle", "3",
            "--num-relax", "0",
            "--random-seed", "42",
            "--save-all",
            str(fasta),
            str(output_dir),
        ]

        print(f"Running {fasta.name}")
        subprocess.run(command, check=True, env=env)


def open_pickle(path):
    if path.name.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            result = pickle.load(f)
    else:
        with path.open("rb") as f:
            result = pickle.load(f)

    if not isinstance(result, dict):
        raise TypeError(f"Invalid result file: {path}")

    return result


def model_seed_token(name):
    match = re.search(r"model_\d+_seed_\d+", name)
    return match.group(0) if match else None


# Pair rank-001 PDB with its corresponding saved result.
def find_result_pair(root, sample_id):
    pdbs = sorted(root.rglob(f"{sample_id}*rank_001*.pdb"))

    pickles = sorted(set(
        list(root.rglob(f"{sample_id}*.pickle"))
        + list(root.rglob(f"{sample_id}*.pickle.gz"))
        + list(root.rglob(f"{sample_id}*.pkl"))
        + list(root.rglob(f"{sample_id}*.pkl.gz"))
    ))

    if not pdbs:
        raise FileNotFoundError(f"No rank-001 PDB: {sample_id}")

    if not pickles:
        raise FileNotFoundError(
            f"No result pickle: {sample_id}. "
            "ColabFold must be run with --save-all."
        )

    pdbs.sort(
        key=lambda x: (
            "unrelaxed" not in x.name,
            "relaxed" not in x.name,
            x.name,
        )
    )

    pdb_path = pdbs[0]
    token = model_seed_token(pdb_path.name)

    if token:
        matched = [p for p in pickles if token in p.name]
    else:
        matched = []

    if len(matched) == 1:
        pickle_path = matched[0]

    elif len(matched) > 1:
        result_files = [p for p in matched if "result" in p.name]

        if len(result_files) == 1:
            pickle_path = result_files[0]
        else:
            raise RuntimeError(
                f"Ambiguous result files for {sample_id}: {matched}"
            )

    elif len(pickles) == 1:
        pickle_path = pickles[0]

    else:
        raise RuntimeError(
            f"Cannot uniquely pair PDB and pickle for {sample_id}"
        )

    return pdb_path, pickle_path, open_pickle(pickle_path)


def softmax_last_axis(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-8, None)


def extract_plddt(result, length):
    if "plddt" not in result:
        raise KeyError("pLDDT was not found in the saved AlphaFold2 result")

    plddt = np.asarray(result["plddt"], dtype=np.float32).reshape(-1)

    if len(plddt) < length:
        raise ValueError(f"Invalid pLDDT length: {len(plddt)}")

    return plddt[:length]


def extract_pae(result, length):
    value = None

    for key in ("predicted_aligned_error", "pae"):
        if key in result:
            value = result[key]
            break

    if value is None:
        raise KeyError(
            "PAE was not found. Use alphafold2_ptm with --save-all."
        )

    if isinstance(value, dict):
        for key in ("predicted_aligned_error", "pae", "value"):
            if key in value:
                value = value[key]
                break

    pae = np.asarray(value, dtype=np.float32)

    if pae.ndim == 3 and pae.shape[0] == 1:
        pae = pae[0]

    if (
        pae.ndim != 2
        or pae.shape[0] < length
        or pae.shape[1] < length
    ):
        raise ValueError(f"Invalid PAE shape: {pae.shape}")

    return pae[:length, :length]


def extract_distogram(result, length):
    logits = None

    if "distogram" in result:
        value = result["distogram"]

        if isinstance(value, dict) and "logits" in value:
            logits = value["logits"]

        elif isinstance(value, np.ndarray):
            logits = value

    if logits is None and "distogram_logits" in result:
        logits = result["distogram_logits"]

    if logits is None:
        raise KeyError(
            "64-bin distogram logits were not found. "
            "The feature cannot be reconstructed from PDB alone."
        )

    logits = np.asarray(logits, dtype=np.float32)

    if logits.ndim == 4 and logits.shape[0] == 1:
        logits = logits[0]

    if (
        logits.ndim != 3
        or logits.shape[-1] != 64
        or logits.shape[0] < length
        or logits.shape[1] < length
    ):
        raise ValueError(f"Invalid distogram shape: {logits.shape}")

    logits = logits[:length, :length, :]
    return softmax_last_axis(logits).astype(np.float32)


def resolve_dssp(executable):
    found = shutil.which(executable)

    if found:
        return found

    candidate = Path(executable)

    if candidate.is_file():
        return str(candidate.resolve())

    for name in ("mkdssp", "dssp"):
        found = shutil.which(name)
        if found:
            return found

    raise FileNotFoundError("DSSP/mkdssp was not found")


# DSSP field 3 is relative solvent accessibility (RSA).
def extract_pdb_dssp(pdb_path, dssp_bin):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("katnet", str(pdb_path))
    model = next(structure.get_models())

    residues = []
    coords = []
    pdb_plddt = []

    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=False) or "CA" not in residue:
                continue

            residues.append((chain.id, residue))
            coords.append(residue["CA"].get_coord())
            pdb_plddt.append(float(residue["CA"].get_bfactor()))

    if not residues:
        raise ValueError(f"No C-alpha residues found: {pdb_path}")

    dssp = DSSP(model, str(pdb_path), dssp=dssp_bin)

    rsa = []
    ss = []

    for chain_id, residue in residues:
        key = (chain_id, residue.id)

        if key not in dssp:
            raise ValueError(
                f"DSSP record missing for chain={chain_id}, residue={residue.id}"
            )

        record = dssp[key]
        code = str(record[2])
        accessibility = float(record[3])

        rsa.append(accessibility)

        if code in {"H", "G", "I"}:
            ss.append([1.0, 0.0, 0.0])

        elif code in {"E", "B"}:
            ss.append([0.0, 1.0, 0.0])

        else:
            ss.append([0.0, 0.0, 1.0])

    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(pdb_plddt, dtype=np.float32),
        np.asarray(rsa, dtype=np.float32),
        np.asarray(ss, dtype=np.float32),
    )


# Verify that PDB and saved AlphaFold result belong to the same prediction.
def verify_pair(pdb_plddt, result_plddt):
    if len(pdb_plddt) != len(result_plddt):
        raise ValueError(
            f"PDB/result length mismatch: {len(pdb_plddt)} vs {len(result_plddt)}"
        )

    mae = float(np.mean(np.abs(pdb_plddt - result_plddt)))

    if mae > 1.0:
        raise ValueError(
            f"PDB/result pLDDT mismatch (MAE={mae:.3f}); "
            "the rank-001 files may not correspond."
        )


def parse_positions(text):
    return np.asarray(
        [int(x) for x in str(text).split(",") if x != ""],
        dtype=np.int64,
    )


# Map ungapped AF2 features back to the original window.
def map_to_original(sequence, positions, coords, plddt, rsa, ss, pae, disto):
    original_length = len(sequence)
    model_length = len(positions)

    expected = {
        "coords": (model_length, 3),
        "plddt": (model_length,),
        "sasa": (model_length,),
        "ss": (model_length, 3),
        "pae": (model_length, model_length),
        "disto": (model_length, model_length, 64),
    }

    actual = {
        "coords": coords.shape,
        "plddt": plddt.shape,
        "sasa": rsa.shape,
        "ss": ss.shape,
        "pae": pae.shape,
        "disto": disto.shape,
    }

    for key in expected:
        if actual[key] != expected[key]:
            raise ValueError(
                f"{key} shape mismatch: {actual[key]} vs {expected[key]}"
            )

    full_coords = np.zeros((original_length, 3), dtype=np.float32)
    full_plddt = np.zeros(original_length, dtype=np.float32)

    # The key remains "sasa" for compatibility; values are DSSP-derived RSA.
    full_sasa = np.zeros(original_length, dtype=np.float32)

    full_ss = np.zeros((original_length, 3), dtype=np.float32)
    full_pae = np.full(
        (original_length, original_length),
        30.0,
        dtype=np.float32,
    )
    full_disto = np.zeros(
        (original_length, original_length, 64),
        dtype=np.float32,
    )

    full_coords[positions] = coords
    full_plddt[positions] = plddt
    full_sasa[positions] = rsa
    full_ss[positions] = ss

    full_pae[np.ix_(positions, positions)] = pae

    full_disto[
        positions[:, None],
        positions[None, :],
        :,
    ] = disto

    return {
        "coords": full_coords,
        "plddt": full_plddt,
        "sasa": full_sasa,
        "ss": full_ss,
        "pae": full_pae,
        "disto": full_disto,
    }


def validate_feature(feature, length):
    expected = {
        "coords": (length, 3),
        "plddt": (length,),
        "sasa": (length,),
        "ss": (length, 3),
        "pae": (length, length),
        "disto": (length, length, 64),
    }

    for key, shape in expected.items():
        if key not in feature:
            raise KeyError(f"Missing feature: {key}")

        if feature[key].shape != shape:
            raise ValueError(
                f"{key}: {feature[key].shape} != {shape}"
            )

        if not np.isfinite(feature[key]).all():
            raise ValueError(f"{key} contains NaN/Inf")


def locate_chunk_output(output_root, sample_id):
    matches = []

    for chunk_dir in sorted(output_root.glob("chunk_*")):
        if list(chunk_dir.rglob(f"{sample_id}*rank_001*.pdb")):
            matches.append(chunk_dir)

    if len(matches) != 1:
        raise RuntimeError(
            f"{sample_id}: expected one ColabFold chunk, found {len(matches)}"
        )

    return matches[0]


# Convert ColabFold outputs to KAT-Net NPZ files.
def convert_features(args):
    manifest_path = args.work_dir / "structure_manifest.csv"

    if not manifest_path.is_file():
        raise FileNotFoundError("Run --stage prepare first")

    manifest = pd.read_csv(manifest_path)

    raw_root = args.work_dir / "colabfold_output"

    if not raw_root.is_dir():
        raise FileNotFoundError("ColabFold output directory was not found")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dssp_bin = resolve_dssp(args.dssp_bin)
    report = []

    for index, row in manifest.iterrows():
        sample_id = str(row["AF2_ID"])
        core_key = str(row["Core_Key"])
        sequence = str(row["Dynamic_Sequence"])
        model_sequence = str(row["Model_Sequence"])
        positions = parse_positions(row["Real_Positions"])

        output_path = args.output_dir / f"{core_key}.npz"

        if output_path.exists() and not args.overwrite:
            report.append({
                "Core_Key": core_key,
                "Status": "existing",
                "Feature_File": str(output_path),
            })
            continue

        print(f"[{index + 1}/{len(manifest)}] {core_key}")

        chunk_dir = locate_chunk_output(raw_root, sample_id)

        pdb_path, pickle_path, result = find_result_pair(
            chunk_dir,
            sample_id,
        )

        coords, pdb_plddt, rsa, ss = extract_pdb_dssp(
            pdb_path,
            dssp_bin,
        )

        length = len(model_sequence)

        if len(coords) != length:
            raise ValueError(
                f"{core_key}: PDB residues={len(coords)}, expected={length}"
            )

        result_plddt = extract_plddt(result, length)
        verify_pair(pdb_plddt, result_plddt)

        pae = extract_pae(result, length)
        disto = extract_distogram(result, length)

        feature = map_to_original(
            sequence,
            positions,
            coords,
            result_plddt,
            rsa,
            ss,
            pae,
            disto,
        )

        validate_feature(feature, len(sequence))

        np.savez_compressed(
            output_path,
            coords=feature["coords"],
            plddt=feature["plddt"],
            sasa=feature["sasa"],
            ss=feature["ss"],
            pae=feature["pae"],
            disto=feature["disto"],
        )

        report.append({
            "Core_Key": core_key,
            "Status": "generated",
            "PDB": str(pdb_path),
            "Result": str(pickle_path),
            "Feature_File": str(output_path),
        })

    report_path = args.work_dir / "feature_manifest.csv"
    pd.DataFrame(report).to_csv(report_path, index=False)

    print(f"\nFeature directory : {args.output_dir}")
    print(f"Feature manifest  : {report_path}")
    print("[Status] PASS")


def check_features(args):
    manifest = pd.read_csv(args.work_dir / "structure_manifest.csv")
    failed = []

    for _, row in manifest.iterrows():
        key = str(row["Core_Key"])
        sequence = str(row["Dynamic_Sequence"])
        path = args.output_dir / f"{key}.npz"

        if not path.is_file():
            failed.append((key, "missing"))
            continue

        try:
            with np.load(path) as data:
                feature = {key: data[key] for key in FEATURE_KEYS}
            validate_feature(feature, len(sequence))

        except Exception as e:
            failed.append((key, str(e)))

    if failed:
        for item in failed[:20]:
            print("FAILED:", item)
        raise RuntimeError(f"{len(failed)} structural feature files failed")

    print(f"Checked {len(manifest):,} feature files")
    print("[Status] PASS")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare AlphaFold2 structural features for KAT-Net"
    )

    parser.add_argument(
        "--stage",
        choices=("prepare", "run", "convert", "check", "all"),
        default="all",
    )

    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="CSV with Core_Key, Dynamic_Sequence and Center_K_Index.",
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("structure_work"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("structure_features"),
    )

    parser.add_argument(
        "--colabfold-bin",
        default="colabfold_batch",
    )

    parser.add_argument(
        "--colabfold-data",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--dssp-bin",
        default="mkdssp",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--gpu",
        default="0",
    )

    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Optional strict window-length check.",
    )

    parser.add_argument(
        "--center-index",
        type=int,
        default=None,
        help="Optional strict center-index check.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.csv_path = args.csv_path.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.colabfold_data is not None:
        args.colabfold_data = args.colabfold_data.expanduser().resolve()

    if not args.csv_path.is_file():
        raise FileNotFoundError(args.csv_path)

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    if args.stage in ("prepare", "all"):
        prepare_manifest(args)

    if args.stage in ("run", "all"):
        run_colabfold(args)

    if args.stage in ("convert", "all"):
        convert_features(args)

    if args.stage in ("check", "all"):
        check_features(args)


if __name__ == "__main__":
    main()