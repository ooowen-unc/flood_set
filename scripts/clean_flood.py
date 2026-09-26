from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTED_EVENT_TYPES = frozenset({"Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood"})
FILE_PATTERN = re.compile(
    r"StormEvents_(details|fatalities|locations)-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv"
)
COUNT_FIELDS = ("INJURIES_DIRECT", "INJURIES_INDIRECT", "DEATHS_DIRECT", "DEATHS_INDIRECT")
REQUIRED = {
    "details": {"EVENT_ID", "EPISODE_ID", "EVENT_TYPE", "YEAR", "STATE", "STATE_FIPS",
                "CZ_TYPE", "CZ_FIPS", "CZ_NAME", "CZ_TIMEZONE", "BEGIN_DATE_TIME",
                "END_DATE_TIME", "DAMAGE_PROPERTY", "DAMAGE_CROPS", *COUNT_FIELDS,
                *(f"{end}_{part}" for end in ("BEGIN", "END")
                  for part in ("YEARMONTH", "DAY", "TIME", "LAT", "LON"))},
    "fatalities": {"EVENT_ID", "FATALITY_ID", "FATALITY_TYPE", "FATALITY_DATE",
                   "FATALITY_AGE", "FATALITY_SEX", "FATALITY_LOCATION"},
    "locations": {"EVENT_ID", "EPISODE_ID", "LOCATION_INDEX", "LATITUDE", "LONGITUDE"},
}
US_STATE_CODES = {
    1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 60, 66, 69, 72, 78,
}
OUTPUT_FIELDS = (
    "event_id", "episode_id", "event_type", "year", "begin_time_local", "end_time_local", "timezone",
    "duration_hours", "state_fips", "county_fips", "area_type", "area_code", "area_name",
    "injuries_direct", "injuries_indirect", "deaths_direct", "deaths_indirect",
    "damage_property_usd", "damage_crops_usd", "flood_cause", "fatality_records",
    "location_points", "year_is_partial", "core_fields_valid", "quality_flags",
)
MAX_COMBINED_BYTES = 1024 ** 3


@dataclass
class Record:
    raw: dict[str, str]
    file: str
    line: int
    year: int
    values: dict = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

def integer(value: str, *, positive: bool = False) -> int | None:
    value = value.strip()
    if not re.fullmatch(r"\d+", value):
        return None
    result = int(value)
    return result if result >= int(positive) else None


def amount(value: str) -> str | None:
    """Accept explicit nonnegative dollar amounts, optionally with K/M/B suffix."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?|\.\d+)\s*([KMB]?)", value.strip().upper())
    if not match:
        return None
    multiplier = {"": 1, "K": 1000, "M": 1000000, "B": 1000000000}[match[2]]
    return format(Decimal(match[1]) * multiplier, "f")


def coordinates(lat: str, lon: str) -> tuple[float | None, float | None]:
    try:
        latitude, longitude = float(lat), float(lon)
    except ValueError:
        return None, None
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return latitude, longitude
    return None, None


def event_datetime(raw: dict[str, str], prefix: str) -> datetime | None:
    """Use four-digit component years, checking the redundant text timestamp."""
    ym = raw[f"{prefix}_YEARMONTH"].strip()
    day = integer(raw[f"{prefix}_DAY"], positive=True)
    time = integer(raw[f"{prefix}_TIME"])
    if not re.fullmatch(r"\d{6}", ym) or day is None or time is None:
        return None
    hour, minute = divmod(time, 100)
    if not (0 <= minute < 60 and (0 <= hour < 24 or time == 2400)):
        return None
    try:
        result = datetime(int(ym[:4]), int(ym[4:]), day) + timedelta(hours=hour, minutes=minute)
    except ValueError:
        return None
    text = raw[f"{prefix}_DATE_TIME"].strip()
    if text:
        # Avoid locale-dependent %b parsing and strptime's two-digit year pivot.
        match = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2}|\d{4}) (\d{2}):(\d{2}):(\d{2})", text)
        if match:
            months = {name: index for index, name in enumerate(
                "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
            text_year = int(match[3])
            year_agrees = text_year == result.year if len(match[3]) == 4 else text_year == result.year % 100
            agrees = (year_agrees and months.get(match[2].upper()) == result.month
                      and int(match[1]) == result.day and int(match[4]) == result.hour
                      and int(match[5]) == result.minute and int(match[6]) == result.second)
        else:
            try:
                agrees = datetime.strptime(text, "%m/%d/%Y %H:%M:%S") == result
            except ValueError:
                agrees = False
        if not agrees:
            return None
    return result


def discover(input_dir: Path, start_year: int, end_year: int) -> tuple[dict, list[str]]:
    """Step 1: choose the newest creation date independently for each table/year."""
    candidates = defaultdict(list)
    for path in input_dir.glob("StormEvents_*.csv"):
        match = FILE_PATTERN.fullmatch(path.name)
        if match and start_year <= int(match[2]) <= end_year:
            candidates[(match[1], int(match[2]))].append((match[3], path))
    chosen, superseded = {}, []
    for year in range(start_year, end_year + 1):
        for kind in REQUIRED:
            options = sorted(candidates[(kind, year)])
            if not options:
                raise ValueError(f"Missing {kind} CSV for {year} in {input_dir}")
            chosen[(kind, year)] = options[-1][1]
            superseded.extend(path.name for _, path in options[:-1])
    return chosen, superseded


def read_records(path: Path, kind: str, year: int):
    # Older NOAA exports contain unescaped quotes inside location/narrative text.
    # Accept their CSV dialect only when the column count remains unchanged;
    # flag selected records that fail strict parsing rather than guessing text.
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = []

        def source_lines():
            for line in stream:
                lines.append(line)
                yield line

        reader = csv.DictReader(source_lines())
        names = reader.fieldnames or []
        missing = REQUIRED[kind] - set(names)
        if missing or len(names) != len(set(names)):
            raise ValueError(f"Invalid {kind} header in {path.name}: missing {sorted(missing)}")
        while True:
            lines.clear()
            start_line = reader.line_num + 1
            try:
                raw = next(reader)
            except StopIteration:
                break
            except csv.Error as error:
                raise ValueError(f"Malformed CSV: {path.name}:{start_line}: {error}") from error
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Column count mismatch: {path.name}:{start_line}")
            record = Record(raw, path.name, start_line, year)
            if kind != "details" or raw["EVENT_TYPE"] in SELECTED_EVENT_TYPES:
                try:
                    list(csv.reader(io.StringIO("".join(lines)), strict=True))
                except csv.Error:
                    record.issues.append("nonstandard_csv_quoting")
            yield record


def unique_records(records, key_fields: tuple[str, ...], kind: str, quarantine: list,
                   counters: Counter) -> list[Record]:
    groups = defaultdict(list)
    for record in records:
        key = tuple(integer(record.raw[name], positive=True) for name in key_fields)
        if None in key:
            quarantine.append((kind, record, "invalid_record_identifier"))
            counters[f"{kind}_invalid_ids"] += 1
        else:
            groups[key].append(record)
    kept = []
    for group in groups.values():
        distinct = {tuple(sorted(record.raw.items())) for record in group}
        if len(distinct) > 1:
            for record in group:
                quarantine.append((kind, record, "conflicting_duplicate_identifier"))
            counters[f"{kind}_conflicting_rows"] += len(group)
        else:
            kept.append(group[0])
            counters[f"{kind}_exact_duplicates_removed"] += len(group) - 1
    return kept


def select_flood(files: dict, quarantine: list, counters: Counter) -> tuple[list, dict]:
    selected, observed_months = [], defaultdict(set)
    for (kind, year), path in sorted(files.items()):
        if kind != "details":
            continue
        print(f"[1/5] Read {path.name}", file=sys.stderr)
        for record in read_records(path, kind, year):
            counters["details_rows_read"] += 1
            ym = record.raw["BEGIN_YEARMONTH"].strip()
            if re.fullmatch(r"\d{6}", ym) and int(ym[:4]) == year and 1 <= int(ym[4:]) <= 12:
                observed_months[year].add(int(ym[4:]))
            if record.raw["EVENT_TYPE"] in SELECTED_EVENT_TYPES:
                selected.append(record)
                counters["flood_rows_selected"] += 1
    return unique_records(selected, ("EVENT_ID",), "details", quarantine, counters), observed_months


def normalize_events(events: list[Record], observed_months: dict) -> None:
    """Step 2: add derived values without overwriting source fields."""
    for record in events:
        raw, values, issues = record.raw, record.values, record.issues
        values["event_id"] = integer(raw["EVENT_ID"], positive=True)
        values["episode_id"] = integer(raw["EPISODE_ID"], positive=True)
        begin, end = event_datetime(raw, "BEGIN"), event_datetime(raw, "END")
        blockers = []
        if values["episode_id"] is None:
            blockers.append("invalid_episode_id")
        if begin is None or end is None or end < begin:
            blockers.append("invalid_or_conflicting_event_time")
        if begin and (begin.year != record.year or integer(raw["YEAR"]) != record.year):
            blockers.append("source_year_conflict")
        timezone = raw["CZ_TIMEZONE"].strip().upper()
        if not re.fullmatch(r"[A-Z]{3,4}(?:[+-]?\d{1,2})?", timezone):
            blockers.append("missing_or_invalid_timezone")
        values.update(begin_time_local=begin.isoformat() if begin else None,
                      end_time_local=end.isoformat() if end else None,
                      duration_hours=(end - begin).total_seconds() / 3600
                      if begin and end and end >= begin else None,
                      event_year=begin.year if begin else None,
                      event_month=begin.month if begin else None,
                      timezone=timezone, event_type=raw["EVENT_TYPE"])
        # Do not invent UTC offsets for historical abbreviations or apply DST rules.
        issues.append("timezone_not_normalized_to_utc")
        state = integer(raw["STATE_FIPS"], positive=True)
        zone = integer(raw["CZ_FIPS"], positive=True)
        area_type = raw["CZ_TYPE"].strip()
        county = None
        if area_type not in {"C", "Z", "M"} or state is None or state > 99 or zone is None or zone > 999:
            blockers.append("invalid_area_code")
            geography = "invalid"
        elif area_type == "C" and state in US_STATE_CODES:
            county = f"{state:02d}{zone:03d}"
            geography = "county_code_unverified"
        else:
            geography = "requires_crosswalk_or_marine_review"
        issues.append(geography)
        values.update(state_fips=f"{state:02d}" if state and state <= 99 else None,
                      area_type=area_type, area_code=f"{zone:03d}" if zone and zone <= 999 else None,
                      county_fips_candidate=county, geography_status=geography)
        for name in COUNT_FIELDS:
            parsed = integer(raw[name])
            values[name.lower()] = parsed
            if parsed is None:
                issues.append(f"{name.lower()}_missing" if not raw[name].strip() else f"{name.lower()}_invalid")
        for name in ("DAMAGE_PROPERTY", "DAMAGE_CROPS"):
            parsed = amount(raw[name])
            values[f"{name.lower()}_usd_nominal"] = parsed
            values[f"{name.lower()}_missing"] = int(parsed is None)
            if parsed is None:
                issues.append(f"{name.lower()}_missing" if not raw[name].strip() else f"{name.lower()}_invalid")
        for prefix in ("BEGIN", "END"):
            lat, lon = coordinates(raw[f"{prefix}_LAT"], raw[f"{prefix}_LON"])
            values[f"{prefix.lower()}_latitude"] = lat
            values[f"{prefix.lower()}_longitude"] = lon
            if lat is None:
                issues.append(f"{prefix.lower()}_coordinates_missing_or_invalid")
        months = sorted(observed_months.get(record.year, set()))
        values.update(source_year_is_partial=int(len(months) != 12),
                      source_year_latest_month=max(months) if months else None,
                      flood_cause=raw.get("FLOOD_CAUSE", "").strip() or None,
                      report_source=raw.get("SOURCE", "").strip() or None,
                      core_valid=int(not blockers))
        issues.extend(blockers)


def link_children(files: dict, events: list[Record], quarantine: list,
                  counters: Counter) -> tuple[list, list]:
    """Step 3: filter first, then deduplicate children without expanding event rows."""
    by_id = {record.values["event_id"]: record for record in events}
    children = {"fatalities": [], "locations": []}
    for (kind, year), path in sorted(files.items()):
        if kind == "details":
            continue
        for record in read_records(path, kind, year):
            counters[f"{kind}_rows_read"] += 1
            event_id = integer(record.raw["EVENT_ID"], positive=True)
            if event_id in by_id:
                record.values["event_id"] = event_id
                children[kind].append(record)
    fatalities = unique_records(children["fatalities"], ("FATALITY_ID",), "fatalities", quarantine, counters)
    locations = unique_records(children["locations"], ("EVENT_ID", "LOCATION_INDEX"), "locations", quarantine, counters)
    accepted_fatalities, accepted_locations = [], []
    for record in fatalities:
        raw = record.raw
        if raw["FATALITY_TYPE"].strip() not in {"D", "I"}:
            quarantine.append(("fatalities", record, "invalid_fatality_type"))
            continue
        record.values.update(fatality_id=integer(raw["FATALITY_ID"], positive=True),
                             fatality_type=raw["FATALITY_TYPE"].strip())
        accepted_fatalities.append(record)
    for record in locations:
        event = by_id[record.values["event_id"]]
        if integer(record.raw["EPISODE_ID"], positive=True) != event.values["episode_id"]:
            quarantine.append(("locations", record, "episode_id_conflict"))
            continue
        lat, lon = coordinates(record.raw["LATITUDE"], record.raw["LONGITUDE"])
        record.values.update(location_index=integer(record.raw["LOCATION_INDEX"], positive=True),
                             latitude=lat, longitude=lon, coordinates_valid=int(lat is not None))
        if lat is None:
            record.issues.append("coordinates_missing_or_invalid")
        accepted_locations.append(record)
    return accepted_fatalities, accepted_locations


def build_features(events: list[Record], fatalities: list[Record], locations: list[Record]) -> list[dict]:
    """Step 4: one event per row; child counts describe records, not total impacts."""
    fatality_counts, location_points = Counter(), defaultdict(list)
    for record in fatalities:
        fatality_counts[(record.values["event_id"], record.values["fatality_type"])] += 1
    for record in locations:
        event_id = record.values["event_id"]
        if record.values["coordinates_valid"]:
            location_points[event_id].append((record.values["location_index"],
                                             record.values["latitude"], record.values["longitude"]))
    features = []
    for record in events:
        values = record.values
        event_id = values["event_id"]
        derived = {name: values.get(name) for name in OUTPUT_FIELDS}
        derived.update(county_fips=values["county_fips_candidate"],
                       damage_property_usd=values["damage_property_usd_nominal"],
                       damage_crops_usd=values["damage_crops_usd_nominal"],
                       area_name=record.raw["CZ_NAME"],
                       year=record.year, core_fields_valid=values["core_valid"],
                       year_is_partial=values["source_year_is_partial"])
        for suffix, code in (("direct", "D"), ("indirect", "I")):
            count = fatality_counts[(event_id, code)]
            reported = values[f"deaths_{suffix}"]
            matches = None if reported is None else int(reported == count)
            if matches == 0:
                record.issues.append(f"deaths_{suffix}_detail_mismatch")
                derived[f"deaths_{suffix}"] = None
        points = [(lat, lon) for _, lat, lon in sorted(location_points[event_id])]
        if not points:
            points = [(values[f"{prefix}_latitude"], values[f"{prefix}_longitude"])
                      for prefix in ("begin", "end") if values[f"{prefix}_latitude"] is not None]
            if points:
                record.issues.append("location_points_from_details")
        points = list(dict.fromkeys(points))
        derived.update(fatality_records=fatality_counts[(event_id, "D")] + fatality_counts[(event_id, "I")],
                       location_points=json.dumps(points, separators=(",", ":")) if points else None,
                       quality_flags="|".join(sorted(set(record.issues))))
        if not points:
            record.issues.append("no_valid_location_points")
            derived["quality_flags"] = "|".join(sorted(set(record.issues)))
        features.append({**derived, "source_year": record.year})
    return features


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def merge_annual_csvs(directory: Path, start_year: int, end_year: int, expected_rows: int) -> dict:
    """Step 5: measure actual files and merge with bounded memory and one header."""
    paths = [directory / f"flood_{year}.csv" for year in range(start_year, end_year + 1)]
    sizes = {path.name: path.stat().st_size for path in paths}
    total_bytes = sum(sizes.values())
    assessment = {"annual_bytes": sizes, "annual_total_bytes": total_bytes,
                  "combined_limit_bytes": MAX_COMBINED_BYTES,
                  "suitable_for_single_csv": total_bytes <= MAX_COMBINED_BYTES}
    if total_bytes > MAX_COMBINED_BYTES:
        return assessment
    combined = directory / f"flood_{start_year}_{end_year}.csv"
    seen = set()
    with combined.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for path in paths:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream, strict=True)
                if reader.fieldnames != list(OUTPUT_FIELDS):
                    raise ValueError(f"Inconsistent annual header: {path.name}")
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(f"Invalid annual row: {path.name}:{reader.line_num}")
                    if row["event_id"] in seen:
                        raise ValueError(f"Duplicate event in annual merge: {row['event_id']}")
                    seen.add(row["event_id"])
                    writer.writerow(row)
    if len(seen) != expected_rows:
        raise ValueError(f"Merged row count {len(seen)} differs from expected {expected_rows}")
    assessment.update(combined_file=combined.name, combined_rows=len(seen),
                      combined_bytes=combined.stat().st_size)
    return assessment


def run(input_dir: Path, output_dir: Path, start_year: int, end_year: int,
        *, dry_run: bool = False) -> dict:
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if output_dir == input_dir or output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise ValueError("Input and output directories must not contain each other")
    if not dry_run and output_dir.exists():
        raise ValueError(f"Output already exists; choose a new output directory: {output_dir}")
    csv.field_size_limit(10_000_000)
    files, superseded = discover(input_dir, start_year, end_year)
    quarantine, counters = [], Counter()
    events, observed_months = select_flood(files, quarantine, counters)
    print("[2/5] Normalize selected flood events", file=sys.stderr)
    normalize_events(events, observed_months)
    print("[3/5] Link fatalities and locations", file=sys.stderr)
    fatalities, locations = link_children(files, events, quarantine, counters)
    print("[4/5] Build event-level features", file=sys.stderr)
    features = build_features(events, fatalities, locations)
    for record in events:
        if not record.values["core_valid"]:
            quarantine.append(("details", record, "invalid_core_fields"))
    flag_counts = Counter(flag for record in events for flag in set(record.issues))
    summary = {
        "event_types": sorted(SELECTED_EVENT_TYPES), "start_year": start_year, "end_year": end_year,
        "dry_run": dry_run, "input_dir": str(input_dir), "output_dir": str(output_dir),
        "selected_source_files": [path.name for path in files.values()],
        "superseded_source_files": superseded, "counts": dict(counters),
        "events_retained": len(events), "feature_rows": len(features),
        "fatalities_retained": len(fatalities), "locations_retained": len(locations),
        "records_requiring_review": len(quarantine), "quality_flags": dict(sorted(flag_counts.items())),
        "flood_events_by_year": dict(sorted(Counter(record.year for record in events).items())),
        "source_observed_months": {year: sorted(observed_months.get(year, set()))
                                   for year in range(start_year, end_year + 1)},
        "review_reasons": dict(Counter(reason for _, _, reason in quarantine)),
        "core_fields_valid_rows": sum(row["core_fields_valid"] for row in features),
        "output_columns": list(OUTPUT_FIELDS),
        "output_files": [f"flood_{year}.csv" for year in range(start_year, end_year + 1)],
        "output_rows_by_year": dict(sorted(Counter(row["source_year"] for row in features).items())),
    }
    if dry_run:
        return summary
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".flood-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        by_year = defaultdict(list)
        for row in features:
            row = dict(row)
            year = row.pop("source_year")
            by_year[year].append(row)
        for year in range(start_year, end_year + 1):
            rows = sorted(by_year[year], key=lambda row: row["event_id"])
            write_csv(stage / f"flood_{year}.csv", rows)
        print("[5/5] Measure annual files and merge", file=sys.stderr)
        summary["storage_assessment"] = merge_annual_csvs(stage, start_year, end_year, len(features))
        combined_file = summary["storage_assessment"].get("combined_file")
        if combined_file:
            summary["output_files"].append(combined_file)
        stage.rename(output_dir)
    return summary


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "Archive")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed_data")
    parser.add_argument("--start-year", type=int, default=1996)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(args.input_dir, args.output_dir, args.start_year, args.end_year, dry_run=args.dry_run)
    except (ValueError, OSError, UnicodeError, csv.Error) as error:
        print(f"Cleaning failed: {error}", file=sys.stderr)
        return 1
    print(f"Selected flood types: {', '.join(summary['event_types'])}")
    print(f"Flood events selected: {summary['counts']['flood_rows_selected']:,}")
    print(f"Output events: {summary['feature_rows']:,}; core fields valid: {summary['core_fields_valid_rows']:,}")
    print(f"Columns: {len(OUTPUT_FIELDS)}; files: {len(summary['output_files'])}")
    for year in range(args.start_year, args.end_year + 1):
        print(f"  {year}: {summary['output_rows_by_year'].get(year, 0):,} events")
    print("Review reasons: " + json.dumps(summary["review_reasons"], sort_keys=True))
    if args.dry_run:
        print("Dry run complete; no files written.")
    else:
        storage = summary["storage_assessment"]
        print(f"Annual total: {storage['annual_total_bytes']:,} bytes")
        if storage["suitable_for_single_csv"]:
            print(f"Combined: {storage['combined_rows']:,} rows, {storage['combined_bytes']:,} bytes")
        else:
            print("Annual files exceed the 1 GiB combined-file budget; annual files retained.")
        print(f"Output: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
