"""Split ALL annual rows without fitting models or learning preprocessing rules.

Uses the standard library only. County grouping is the default; row shuffling
is optional. Raw columns, blanks, leading zeros and original row order survive.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter
from contextlib import ExitStack
from decimal import Decimal
from pathlib import Path

from config import KEYS, LABELS, REQUIRED, SOURCE, SPLITS, TARGET


def partition(keys, fractions, seed):
    keys = sorted(keys)
    random.Random(seed).shuffle(keys)
    train_end = int(len(keys) * fractions[0])
    validation_end = train_end + int(len(keys) * fractions[1])
    if not 0 < train_end < validation_end < len(keys):
        raise ValueError("Not enough groups for three nonempty splits")
    return {key: name for name, subset in (
        ("train", keys[:train_end]), ("validation", keys[train_end:validation_end]),
        ("test", keys[validation_end:])) for key in subset}


def split_data(source, output_dir, *, unit="county", fractions=(.7, .15, .15), seed=42, overwrite=False):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if any(not math.isfinite(f) or f <= 0 for f in fractions) or not math.isclose(sum(fractions), 1):
        raise ValueError("Three positive fractions must sum to 1")
    if unit not in {"county", "row"}:
        raise ValueError("unit must be county or row")
    outputs = [output_dir / (name + ".csv") for name in ("train", "validation", "test")]
    outputs.append(output_dir / "split_metadata.json")
    if any(path == source or (path.exists() and not overwrite) for path in outputs):
        raise ValueError("Input/existing output would be overwritten; --overwrite replaces split outputs only")
    seen, groups, count = set(), set(), 0
    csv.field_size_limit(10_000_000)
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, strict=True)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or set(REQUIRED) - set(fields):
            raise ValueError("Source header is incomplete or duplicated")
        for i, row in enumerate(reader):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"Column mismatch on source line {reader.line_num}")
            cf, year = row["county_fips"], row["analysis_year"]
            if not re.fullmatch(r"\d{5}", cf) or not re.fullmatch(r"\d{4}", year) or not 1999 <= int(year) <= 2025:
                raise ValueError(f"Invalid county/year: {cf}/{year}")
            if (cf, year) in seen:
                raise ValueError(f"Repeated county/year: {cf}/{year}")
            seen.add((cf, year))
            groups.add(cf if unit == "county" else i)
            count += 1
    assignments = partition(groups, fractions, seed)
    summaries = {name: Counter() for name in ("train", "validation", "test")}
    county_sets = {name: set() for name in summaries}
    output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        writers = {}
        for name, path in zip(summaries, outputs):
            stream = stack.enter_context(path.open("w", encoding="utf-8-sig", newline=""))
            writers[name] = csv.DictWriter(stream, fieldnames=fields)
            writers[name].writeheader()
        f = stack.enter_context(source.open(encoding="utf-8-sig", newline=""))
        for i, row in enumerate(csv.DictReader(f, strict=True)):
            name = assignments[row["county_fips"] if unit == "county" else i]
            writers[name].writerow(row)
            summary = summaries[name]
            summary["rows"] += 1
            summary["year:" + row["analysis_year"]] += 1
            county_sets[name].add(row["county_fips"])
            if row[TARGET].strip():
                summary["positive_pa_target_rows"] += Decimal(row[TARGET]) > 0
            summary["four_measure_labeled_rows"] += any(Decimal(row[f] or "0") > 0 for f in LABELS)
            for label in LABELS:
                summary[label + "_positive_rows"] += Decimal(row[label] or "0") > 0
    if sum(s["rows"] for s in summaries.values()) != count:
        raise ValueError("Split row counts do not reconcile")
    if unit == "county" and any(county_sets[a] & county_sets[b] for a, b in
                                 (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("County overlap across splits")
    with source.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    metadata = {"source": str(source), "source_sha256": digest, "seed": seed,
                "unit": unit, "requested_fractions": dict(zip(summaries, fractions)),
                "source_rows": count, "columns": len(fields),
                "splits": {name: {"counties": len(county_sets[name]), "actual_row_fraction": s["rows"] / count,
                                  **dict(s)} for name, s in summaries.items()},
                "notes": ["All original county/year rows and columns retained; no labels or features imputed.",
                          "County grouping randomizes entire counties; row proportions are approximate.",
                          "Random splits assess retrospective association, not forecasting future years.",
                          "Impact thresholds, imputers, scaling and models are NOT fitted by this script."]}
    outputs[3].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=SPLITS)
    parser.add_argument("--unit", choices=("county", "row"), default="county")
    parser.add_argument("--fractions", nargs=3, type=float, default=(.7, .15, .15), metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    meta = split_data(args.source, args.output_dir, unit=args.unit, fractions=args.fractions,
                      seed=args.seed, overwrite=args.overwrite)
    print(json.dumps({"unit": meta["unit"], "seed": meta["seed"],
                      "rows": {name: info["rows"] for name, info in meta["splits"].items()},
                      "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
