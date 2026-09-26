from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EST = timezone(timedelta(hours=-5), "EST")
# Numeric suffixes must agree with their named zone, rather than override it.
UTC_OFFSETS = {
    "EST": -5, "CST": -6, "MST": -7, "PST": -8, "AKST": -9,
    "HST": -10, "AST": -4, "SST": -11, "GST": 10,
    "EDT": -4, "CDT": -5, "MDT": -6, "PDT": -7, "AKDT": -8, "HDT": -9,
}
REQUIRED_FIELDS = {
    "event_id", "event_type", "year", "begin_time_local", "end_time_local", "timezone",
    "duration_hours", "core_fields_valid", "damage_property_usd", "damage_crops_usd",
}
REPLACEMENTS = {"begin_time_local": "begin_time_est", "end_time_local": "end_time_est"}


def source_zone(label: str) -> timezone:
    normalized = label.strip().upper()
    match = re.fullmatch(r"([A-Z]{3,4})([+-]?\d{1,2})?", normalized)
    if not match or match[1] not in UTC_OFFSETS:
        raise ValueError(f"Unknown NOAA source timezone: {label!r}")
    hours = UTC_OFFSETS[match[1]]
    if match[2] is not None and int(match[2]) != hours:
        raise ValueError(f"Conflicting timezone name and offset: {label!r}")
    return timezone(timedelta(hours=hours), normalized)


def local_time(value: str, field: str) -> datetime:
    # Require a timestamp, not a date-only value. An aware input is a different schema.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?", value):
        raise ValueError(f"Invalid naive source timestamp in {field}: {value!r}")
    return datetime.fromisoformat(value)


def convert_row(row: dict[str, str]) -> tuple[dict[str, str], bool, bool]:
    zone = source_zone(row["timezone"])
    begin = local_time(row["begin_time_local"], "begin_time_local")
    end = local_time(row["end_time_local"], "end_time_local")
    begin_est = begin.replace(tzinfo=zone).astimezone(EST)
    end_est = end.replace(tzinfo=zone).astimezone(EST)
    seconds = (end_est - begin_est).total_seconds()
    if seconds < 0:
        raise ValueError("End timestamp precedes begin timestamp")
    duration = row["duration_hours"].strip()
    if duration:
        try:
            provided = Decimal(duration)
        except InvalidOperation as error:
            raise ValueError(f"Invalid duration_hours: {duration!r}") from error
        expected = Decimal(str(seconds)) / Decimal(3600)
        if not provided.is_finite() or abs(provided - expected) > Decimal("0.000000001"):
            raise ValueError(f"duration_hours disagrees with timestamps: {duration!r}")
    output = {REPLACEMENTS.get(key, key): value for key, value in row.items()
              if key != "quality_flags"}
    output.update(begin_time_est=begin_est.isoformat(), end_time_est=end_est.isoformat(),
                  timezone="EST-5", source_timezone=row["timezone"])
    date_changed = begin.date() != begin_est.date() or end.date() != end_est.date()
    year_changed = begin.year != begin_est.year or end.year != end_est.year
    return output, date_changed, year_changed


def clean_file(input_path: Path, output_path: Path, *, overwrite: bool = False) -> dict:
    """Filter and stream source events to EST; publish only after complete success."""
    input_path, output_path = input_path.resolve(), output_path.resolve()
    if input_path == output_path:
        raise ValueError("Input and output must be different files")
    if output_path.exists() and not overwrite:
        raise ValueError(f"Output already exists: {output_path}")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"Output is not a regular file: {output_path}")
    if not input_path.is_file():
        raise ValueError(f"Input file does not exist: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen, zones, counts = set(), Counter(), Counter()
    date_changes = year_changes = 0
    csv.field_size_limit(10_000_000)
    with tempfile.TemporaryDirectory(prefix=".flood-est-", dir=output_path.parent) as temporary:
        staged = Path(temporary) / output_path.name
        with input_path.open(encoding="utf-8-sig", newline="") as source, staged.open(
            "w", encoding="utf-8-sig", newline=""
        ) as target:
            reader = csv.DictReader(source, strict=True)
            fields = reader.fieldnames or []
            missing = REQUIRED_FIELDS - set(fields)
            reserved = {"begin_time_est", "end_time_est", "source_timezone"} & set(fields)
            if missing or reserved or len(fields) != len(set(fields)):
                raise ValueError(f"Invalid source header: missing {sorted(missing)}, reserved {sorted(reserved)}")
            output_fields = [REPLACEMENTS.get(name, name) for name in fields if name != "quality_flags"]
            output_fields.insert(output_fields.index("timezone") + 1, "source_timezone")
            writer = csv.DictWriter(target, fieldnames=output_fields)
            writer.writeheader()
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"CSV column count mismatch at line {reader.line_num}")
                event_id = row["event_id"]
                if not re.fullmatch(r"[1-9]\d*", event_id) or event_id in seen:
                    raise ValueError(f"Invalid or duplicate event_id at line {reader.line_num}: {event_id!r}")
                seen.add(event_id)
                counts["input_rows"] += 1
                year_text = row["year"].strip()
                if not re.fullmatch(r"\d{4}", year_text):
                    raise ValueError(f"Event {event_id}: invalid source year {year_text!r}")
                if not 1998 <= int(year_text) <= 2025:
                    counts["removed_year_outside_range"] += 1
                    continue
                valid = row["core_fields_valid"].strip()
                if valid not in {"0", "1"}:
                    raise ValueError(f"Event {event_id}: invalid core_fields_valid {valid!r}")
                if valid == "0":
                    counts["removed_core_invalid"] += 1
                    continue
                property_damage = row["damage_property_usd"].strip()
                crop_damage = row["damage_crops_usd"].strip()
                if not property_damage and not crop_damage:
                    counts["removed_both_damage_missing"] += 1
                    continue
                if not property_damage:
                    row["damage_property_usd"] = "0"
                    counts["filled_property_damage"] += 1
                if not crop_damage:
                    row["damage_crops_usd"] = "0"
                    counts["filled_crop_damage"] += 1
                try:
                    converted, date_changed, year_changed = convert_row(row)
                except (ValueError, OverflowError) as error:
                    raise ValueError(f"Event {event_id}, line {reader.line_num}: {error}") from error
                writer.writerow(converted)
                counts["output_rows"] += 1
                zones[row["timezone"]] += 1
                date_changes += int(date_changed)
                year_changes += int(year_changed)
        # A failing conversion leaves the source untouched and publishes no partial CSV.
        if overwrite:
            staged.replace(output_path)
        else:
            staged.rename(output_path)
    return {"rows": counts["output_rows"], "input_rows": counts["input_rows"],
            "removed_year_outside_range": counts["removed_year_outside_range"],
            "removed_core_invalid": counts["removed_core_invalid"],
            "removed_both_damage_missing": counts["removed_both_damage_missing"],
            "filled_property_damage": counts["filled_property_damage"],
            "filled_crop_damage": counts["filled_crop_damage"],
            "source_timezones": dict(sorted(zones.items())),
            "events_with_date_change": date_changes, "events_with_year_change": year_changes,
            "columns": len(output_fields), "output_bytes": output_path.stat().st_size,
            "output_path": str(output_path)}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed_data/flood_1996_2026.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed_data/flood_final.csv")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output only after successful cleaning")
    args = parser.parse_args()
    try:
        summary = clean_file(args.input, args.output, overwrite=args.overwrite)
    except (ValueError, OSError, csv.Error) as error:
        print(f"Final cleaning failed: {error}", file=sys.stderr)
        return 1
    print(f"Input: {summary['input_rows']:,}; retained: {summary['rows']:,} events, in EST (UTC-05:00).")
    print(f"Removed source years outside 1998-2025: {summary['removed_year_outside_range']:,}")
    print(f"Removed core_fields_valid=0: {summary['removed_core_invalid']:,}")
    print(f"Removed both damage amounts missing after core filter: {summary['removed_both_damage_missing']:,}")
    print(f"Filled property damage: {summary['filled_property_damage']:,}; crop damage: {summary['filled_crop_damage']:,}")
    print(f"Source timezone labels: {len(summary['source_timezones'])}")
    print(f"Events crossing a date boundary: {summary['events_with_date_change']:,}")
    print(f"Events crossing a year boundary: {summary['events_with_year_change']:,}")
    print(f"Output: {summary['output_path']}; {summary['columns']} columns; {summary['output_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
