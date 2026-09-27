from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from collections import Counter
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAMAGE_FIELDS = ("damage_property_usd", "damage_crops_usd")
METADATA_FIELDS = ("cpi_base_month", "cpi_interpolated")
CENT = Decimal("0.01")
EST_OFFSET = timedelta(hours=-5)


def parse_month(value: str) -> date:
    if not re.fullmatch(r"\d{4}-\d{2}", value):
        raise ValueError(f"Expected YYYY-MM month, got {value!r}")
    return date.fromisoformat(value + "-01")


def positive_cpi(value: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"Invalid CPI value: {value!r}") from error
    if not result.is_finite() or result <= 0:
        raise ValueError(f"CPI must be finite and positive: {value!r}")
    return result


def read_cpi(path: Path) -> dict[date, Decimal | None]:
    values = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        fields = reader.fieldnames or []
        if not {"observation_date", "CPIAUCSL"} <= set(fields) or len(fields) != len(set(fields)):
            raise ValueError("CPI header must contain unique observation_date and CPIAUCSL columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CPI column count mismatch at line {reader.line_num}")
            observed = date.fromisoformat(row["observation_date"].strip())
            if observed.day != 1 or observed in values:
                raise ValueError(f"Invalid or duplicate CPI month at line {reader.line_num}")
            text = row["CPIAUCSL"].strip()
            values[observed] = None if text in {"", "."} else positive_cpi(text)
    return values


def month_cpi(values: dict[date, Decimal | None], month: date, *, interpolate: bool) -> tuple[Decimal, bool]:
    observed = values.get(month)
    if observed is not None:
        return observed, False
    if interpolate:
        previous = (month - timedelta(days=1)).replace(day=1)
        following = (month + timedelta(days=32)).replace(day=1)
        left, right = values.get(previous), values.get(following)
        if left is not None and right is not None:
            return (left + right) / Decimal(2), True
    raise ValueError(f"Missing CPI for {month:%Y-%m}; interpolation requires both adjacent observed months")


def event_month(timestamp: str) -> date:
    if "T" not in timestamp:
        raise ValueError(f"Invalid begin_time_est: {timestamp!r}")
    begin = datetime.fromisoformat(timestamp)
    if begin.utcoffset() != EST_OFFSET:
        raise ValueError(f"begin_time_est must have a fixed -05:00 offset: {timestamp!r}")
    return date(begin.year, begin.month, 1)


def adjust_amount(value: str, base_cpi: Decimal, source_cpi: Decimal) -> str:
    if not value.strip():
        return value
    try:
        amount = Decimal(value.strip())
    except InvalidOperation as error:
        raise ValueError(f"Invalid damage amount: {value!r}") from error
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"Damage must be finite and nonnegative: {value!r}")
    # Do not round the CPI ratio before applying it to an amount.
    return format((amount * base_cpi / source_cpi).quantize(CENT, rounding=ROUND_HALF_UP), "f")


def adjust_file(input_path: Path, cpi_path: Path, output_path: Path, *,
                base_month: str = "2026-08", missing_cpi: str = "interpolate", overwrite: bool = False) -> dict:
    input_path, cpi_path, output_path = input_path.resolve(), cpi_path.resolve(), output_path.resolve()
    if output_path in {input_path, cpi_path}:
        raise ValueError("Output must differ from both input files")
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output_path}; use --overwrite to replace it")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"Output is not a regular file: {output_path}")
    if missing_cpi not in {"error", "interpolate"}:
        raise ValueError(f"Unknown missing CPI policy: {missing_cpi!r}")
    values = read_cpi(cpi_path)
    # Never interpolate the target price level itself.
    base_value, _ = month_cpi(values, parse_month(base_month), interpolate=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    estimated_months = Counter()
    rows = 0
    factors = {}
    csv.field_size_limit(10_000_000)
    # A private TemporaryDirectory on Windows can give its children sandbox-only
    # ACLs that survive rename. A file in the destination inherits its normal ACL.
    with tempfile.NamedTemporaryFile(prefix=".flood-cpi-", suffix=".csv",
                                     dir=output_path.parent, delete=False) as temporary:
        staged = Path(temporary.name)
    # Close the temporary handle before reopening/replacing it on Windows.
    try:
        with input_path.open(encoding="utf-8-sig", newline="") as source, staged.open(
            "w", encoding="utf-8-sig", newline=""
        ) as target:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames or []
            missing = {"begin_time_est", *DAMAGE_FIELDS} - set(fields)
            reserved = set(METADATA_FIELDS) & set(fields)
            if missing or reserved or len(fields) != len(set(fields)):
                raise ValueError(f"Invalid flood header: missing {sorted(missing)}, reserved {sorted(reserved)}")
            writer = csv.DictWriter(target, fieldnames=[*fields, *METADATA_FIELDS])
            writer.writeheader()
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Flood column count mismatch at line {reader.line_num}")
                try:
                    month = event_month(row["begin_time_est"])
                    if month not in factors:
                        factors[month] = month_cpi(values, month, interpolate=missing_cpi == "interpolate")
                    source_value, estimated = factors[month]
                    for field in DAMAGE_FIELDS:
                        row[field] = adjust_amount(row[field], base_value, source_value)
                except (ValueError, InvalidOperation, OverflowError) as error:
                    raise ValueError(f"Event {row.get('event_id', '?')}, line {reader.line_num}: {error}") from error
                row.update(cpi_base_month=base_month, cpi_interpolated=str(int(estimated)))
                writer.writerow(row)
                rows += 1
                if estimated:
                    estimated_months[f"{month:%Y-%m}"] += 1
        if overwrite:
            staged.replace(output_path)
        else:
            staged.rename(output_path)
    finally:
        staged.unlink(missing_ok=True)
    return {"rows": rows, "base_month": base_month, "base_cpi": str(base_value),
            "interpolated_months": dict(sorted(estimated_months.items())),
            "interpolated_values": {month: str(factors[parse_month(month)][0]) for month in sorted(estimated_months)},
            "output_path": str(output_path)}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_data/flood_final.csv")
    parser.add_argument("--cpi", type=Path, default=ROOT / "CPIAUCSL.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed_data/flood_final_CPI.csv")
    parser.add_argument("--base-month", default="2026-08", help="Price basis month, YYYY-MM")
    parser.add_argument("--missing-cpi", choices=("error", "interpolate"), default="interpolate",
                        help="Missing CPI policy; by default, interpolate only isolated missing months using adjacent observations")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output after all processing succeeds")
    args = parser.parse_args()
    try:
        summary = adjust_file(args.input, args.cpi, args.output, base_month=args.base_month,
                              missing_cpi=args.missing_cpi, overwrite=args.overwrite)
    except (ValueError, OSError, csv.Error, InvalidOperation) as error:
        print(f"CPI adjustment failed: {error}", file=sys.stderr)
        return 1
    print(f"Adjustment complete: {summary['rows']:,} rows; price basis {summary['base_month']}, CPI={summary['base_cpi']}")
    for month, count in summary["interpolated_months"].items():
        print(f"Interpolated CPI: {month}={summary['interpolated_values'][month]}, used for {count:,} rows (cpi_interpolated=1)")
    print(f"Output: {summary['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
