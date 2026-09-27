from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

try:
    from .adjust_flood_cpi import adjust_amount, event_month, month_cpi, parse_month, read_cpi
except ImportError:
    from adjust_flood_cpi import adjust_amount, event_month, month_cpi, parse_month, read_cpi


ROOT = Path(__file__).resolve().parents[1]
PRICE_SOURCE = "https://www.arcgis.com/home/item.html?id=1cb56c682f6f4ce08a07ff372a7908b0"
EVENT_HAZARDS = {"Flood": "IFLD", "Flash Flood": "IFLD", "Coastal Flood": "CFLD",
                 "Lakeshore Flood": ""}
COMMON_FIELDS = ("NRI_ID", "STATE", "STATEABBRV", "COUNTY", "COUNTYTYPE", "POPULATION",
                 "BUILDVALUE", "AGRIVALUE", "AREA", "SOVI_SCORE", "SOVI_RATNG",
                 "RESL_SCORE", "RESL_RATNG", "RESL_VALUE", "CRF_VALUE", "NRI_VER")
HAZARD_SUFFIXES = ("EVNTS", "AFREQ", "EXP_AREA", "EXPB", "EXPP", "EXPPE", "EXPA",
                   "EXPT", "HLRB", "HLRP", "HLRA", "HLRR", "EALB", "EALP", "EALPE",
                   "EALA", "EALT", "EALS", "EALR", "ALRB", "ALRP", "ALRA",
                   "ALR_NPCTL", "RISKS", "RISKR")
# EALP and EXPP are people, not dollars. RISKV is an index value, not observed loss.
MONEY_SUFFIXES = {"EXPB", "EXPPE", "EXPA", "EXPT", "EALB", "EALPE", "EALA", "EALT"}
COMMON_MONEY = {"BUILDVALUE", "AGRIVALUE"}
# Keep high-use predictors inline; the county/hazard table holds the complete snapshot.
EVENT_CONTEXT_FIELDS = (
    "nri_id", "nri_population", "nri_sovi_score", "nri_resl_score", "nri_risks", "nri_riskr",
    "nri_buildvalue_adjusted_usd", "nri_agrivalue_adjusted_usd", "nri_ealb_adjusted_usd",
    "nri_eala_adjusted_usd", "nri_material_eal_adjusted_usd", "nri_ver",
)
COUNTS = ("injuries_direct", "injuries_indirect", "deaths_direct", "deaths_indirect")
LOSSES = ("property", "crops", "material")
OUTPUT_NAMES = ("flood_nri_events.csv", "nri_flood_counties.csv", "flood_nri_county_year.csv",
                "data_dictionary.csv", "analysis_metadata.json")
EVENT_DERIVED = (
    "analysis_year", "event_month", "flood_category", "nri_hazard_prefix", "nri_join_status",
    "nri_hazard_mapping_status", "source_year_differs_from_analysis_year",
    "within_nri_hazard_record_period", "cpi_source_month", "cpi_source_value",
    "cpi_target_value", "cpi_factor", "cpi_interpolated", "cpi_base_month",
    "damage_property_adjusted_usd", "damage_crops_adjusted_usd",
    "damage_material_original_usd", "damage_material_adjusted_usd",
)


def csv_rows(path: Path, required: set[str] | None = None):
    """Strict row reader; strings preserve leading zeros and source missingness."""
    csv.field_size_limit(10_000_000)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        names = reader.fieldnames or []
        if len(names) != len(set(names)) or not names or (required or set()) - set(names):
            raise ValueError(f"Invalid or incomplete CSV header: {path}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Column count mismatch: {path}, line {reader.line_num}")
            yield row


def number(value: str) -> Decimal | None:
    if not value.strip():
        return None
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError(f"Expected finite nonnegative number: {value!r}")
    return result


def ratio(numerator: Decimal | int | None, denominator: Decimal | None, scale=1) -> str:
    if numerator is None or denominator is None or denominator <= 0:
        return ""
    return format((Decimal(numerator) * scale / denominator).quantize(Decimal("0.00000001")), "f")


def currency(value: Decimal | None) -> str:
    return "" if value is None else format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")


def fips_key(value: str) -> str:
    if not re.fullmatch(r"\d{5}", value):
        raise ValueError(f"Expected five-character county FIPS: {value!r}")
    return value


def read_counties(path: Path) -> dict[str, dict[str, str]]:
    counties = {}
    for row in csv_rows(path, {"STCOFIPS"}):
        key = fips_key(row["STCOFIPS"])
        if key in counties:
            raise ValueError(f"Duplicate county FIPS in {path}: {key}")
        counties[key] = row
    if not counties:
        raise ValueError(f"Empty NRI table: {path}")
    return counties


def reconcile_cleaned(full: dict, cleaned: dict) -> dict:
    """Allow float export rounding, but expose missing-to-zero and other changes."""
    differences = Counter()
    compared = 0
    for key in full.keys() & cleaned.keys():
        relevant = set(COMMON_FIELDS) | {p + "_" + s for p in ("IFLD", "CFLD") for s in HAZARD_SUFFIXES}
        for name in full[key].keys() & cleaned[key].keys() & relevant:
            left, right = full[key][name].strip(), cleaned[key][name].strip()
            compared += 1
            if left == right:
                continue
            try:
                a, b = Decimal(left), Decimal(right)
                equivalent = a.is_finite() and b.is_finite() and abs(a - b) <= max(
                    Decimal("0.00000001"), abs(a) * Decimal("0.000000000001"))
            except InvalidOperation:
                equivalent = False
            if not equivalent:
                differences[name] += 1
    return {"authoritative_source": "NRI_Table_Counties.csv", "compared_cells": compared,
            "full_only_fips": sorted(full.keys() - cleaned.keys()),
            "cleaned_only_fips": sorted(cleaned.keys() - full.keys()),
            "different_cells_by_field": dict(sorted(differences.items())),
            "policy": "Use full-table values, including blanks; never replace them with cleaned zeros."}


def common_column(name: str) -> str:
    return "nri_id" if name == "NRI_ID" else "nri_" + name.lower()


def nri_columns() -> list[str]:
    columns = [common_column(name) for name in COMMON_FIELDS]
    columns += ["nri_" + name.lower() + "_adjusted_usd" for name in sorted(COMMON_MONEY)]
    for suffix in HAZARD_SUFFIXES:
        columns.append("nri_" + suffix.lower())
        if suffix in MONEY_SUFFIXES:
            columns.append("nri_" + suffix.lower() + "_adjusted_usd")
    columns += ["nri_material_eal_adjusted_usd", "nri_price_source_month", "nri_cpi_factor",
                "nri_price_target_month", "nri_hazard_name", "nri_record_start",
                "nri_record_end", "nri_record_years", "nri_frequency_model",
                "nri_record_description"]
    return columns


def make_context(county: dict, prefix: str, hazard: dict, base_cpi: Decimal,
                 nri_cpi: Decimal, nri_month: str, base_month: str) -> dict:
    context = {common_column(name): county[name] for name in COMMON_FIELDS}
    for name in sorted(COMMON_MONEY):
        context["nri_" + name.lower() + "_adjusted_usd"] = adjust_amount(county[name], base_cpi, nri_cpi)
    for suffix in HAZARD_SUFFIXES:
        value = county.get(prefix + "_" + suffix, "") if prefix else ""
        context["nri_" + suffix.lower()] = value
        if suffix in MONEY_SUFFIXES:
            context["nri_" + suffix.lower() + "_adjusted_usd"] = adjust_amount(value, base_cpi, nri_cpi)
    building = number(context["nri_ealb_adjusted_usd"])
    agriculture = number(context["nri_eala_adjusted_usd"])
    # Coastal EAL does not model agriculture; it is not a zero agriculture estimate.
    context["nri_material_eal_adjusted_usd"] = currency(
        building + agriculture if building is not None and agriculture is not None else None)
    context.update(nri_price_source_month=nri_month, nri_price_target_month=base_month,
                   nri_cpi_factor=str(base_cpi / nri_cpi), nri_hazard_name=hazard.get("Hazard", ""),
                   nri_record_start=hazard.get("Start", ""), nri_record_end=hazard.get("End", ""),
                   nri_record_years=hazard.get("TotalYears", ""),
                   nri_frequency_model=hazard.get("FrequencyModel", ""),
                   nri_record_description=hazard.get("PeriodOfRecord", ""))
    return context


@dataclass
class AnnualGroup:
    events: int = 0
    episodes: set[str] = field(default_factory=set)
    missing_episodes: int = 0
    event_types: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    missing_counts: Counter = field(default_factory=Counter)
    sums: Counter = field(default_factory=Counter)
    known_losses: Counter = field(default_factory=Counter)
    max_material: Decimal = Decimal(0)
    interpolated_events: int = 0
    partial_year: int = 0

    def add(self, row: dict):
        self.events += 1
        self.event_types[row["event_type"]] += 1
        if row["episode_id"]:
            self.episodes.add(row["episode_id"])
        else:
            self.missing_episodes += 1
        self.interpolated_events += int(row["cpi_interpolated"])
        self.partial_year = max(self.partial_year, int(row["year_is_partial"]))
        for name in COUNTS:
            value = number(row[name])
            if value is None:
                self.missing_counts[name] += 1
            else:
                if value != value.to_integral_value():
                    raise ValueError(f"Noninteger casualty count: {name}={value}")
                self.counts[name] += int(value)
        for basis in LOSSES:
            original = row["damage_" + basis + "_usd"] if basis != "material" else row["damage_material_original_usd"]
            adjusted = row["damage_" + basis + "_adjusted_usd"]
            if adjusted:
                self.known_losses[basis] += 1
                self.sums[basis + "_original"] += Decimal(original)
                self.sums[basis + "_adjusted"] += Decimal(adjusted)
                if basis == "material":
                    self.max_material = max(self.max_material, Decimal(adjusted))

    def result(self, context: dict) -> dict:
        result = {"event_records": self.events, "distinct_episode_ids": len(self.episodes),
                  "missing_episode_records": self.missing_episodes,
                  "flash_flood_records": self.event_types["Flash Flood"],
                  "flash_flood_record_share": ratio(self.event_types["Flash Flood"], Decimal(self.events)),
                  "cpi_interpolated_records": self.interpolated_events,
                  "year_is_partial": self.partial_year}
        for name in COUNTS:
            result[name] = self.counts[name] if self.missing_counts[name] < self.events else ""
            result[name + "_missing_records"] = self.missing_counts[name]
        for basis in LOSSES:
            known = self.known_losses[basis]
            result[basis + "_loss_known_records"] = known
            result[basis + "_loss_original_usd"] = currency(self.sums[basis + "_original"] if known else None)
            result[basis + "_loss_adjusted_usd"] = currency(self.sums[basis + "_adjusted"] if known else None)
        pop = number(context["nri_population"])
        material = self.sums["material_adjusted"] if self.known_losses["material"] == self.events else None
        prop = self.sums["property_adjusted"] if self.known_losses["property"] == self.events else None
        crops = self.sums["crops_adjusted"] if self.known_losses["crops"] == self.events else None
        deaths = self.counts["deaths_direct"] if not self.missing_counts["deaths_direct"] else None
        result.update(
            material_loss_per_nri_population_usd=ratio(material, pop),
            direct_deaths_per_100k_nri_population=ratio(deaths, pop, 100000),
            property_loss_to_nri_buildvalue_ratio=ratio(prop, number(context["nri_buildvalue_adjusted_usd"])),
            crops_loss_to_nri_agrivalue_ratio=ratio(crops, number(context["nri_agrivalue_adjusted_usd"])),
            property_loss_to_nri_building_eal_ratio=ratio(prop, number(context["nri_ealb_adjusted_usd"])),
            crops_loss_to_nri_agriculture_eal_ratio=ratio(crops, number(context["nri_eala_adjusted_usd"])),
            material_loss_to_nri_material_eal_ratio=ratio(material, number(context["nri_material_eal_adjusted_usd"])),
            largest_event_material_loss_share=ratio(self.max_material, material),
        )
        return result


def join_status(row: dict, counties: dict) -> str:
    if row["area_type"] != "C":
        return "unmatched_forecast_zone" if row["area_type"] == "Z" else "unmatched_noncounty_area"
    key = row["county_fips"]
    if not key:
        return "unmatched_missing_county_fips"
    fips_key(key)
    if key[:2] != row["state_fips"]:
        raise ValueError(f"State/county FIPS disagree in event {row['event_id']}")
    if key in counties:
        return "matched_county"
    return "unmatched_connecticut_boundary" if key.startswith("09") else "unmatched_county_fips"


def record_period_flag(context: dict, year: int) -> str:
    # Probability model dates are dataset dates, not an observed-event comparison window.
    if context.get("nri_frequency_model") != "Annualized Frequency":
        return ""
    return str(int(int(context["nri_record_start"]) <= year <= int(context["nri_record_end"])))


def annual_columns() -> list[str]:
    dummy = {name: "" for name in nri_columns()}
    return ["county_fips", "analysis_year", "flood_category", "nri_hazard_prefix",
            "within_nri_hazard_record_period", "cpi_base_month", *AnnualGroup().result(dummy), *nri_columns()]


def write_csv(path: Path, columns: list[str], rows):
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def field_dictionary(schemas: dict, nri_dictionary: dict) -> list[dict]:
    descriptions = {
        "analysis_year": "Year of begin_time_est (fixed UTC-05:00); aggregation year.",
        "event_month": "Month of begin_time_est; entire event loss assigned to its start month.",
        "cpi_factor": "CPI target month / CPI event start month; unrounded Decimal calculation.",
        "cpi_interpolated": "1 only when missing source-month CPI was replaced by the mean of adjacent observed months.",
        "nri_join_status": "matched_county or unmatched reason; no geography is guessed or duplicated.",
        "nri_hazard_mapping_status": "mapped or unmapped_lakeshore; mapping and geographic matching are independent.",
        "within_nri_hazard_record_period": "0/1 for annualized-frequency hazards only; blank for probability/unmapped hazards.",
        "event_records": "Retained flood_final records, not complete flood occurrence frequency.",
        "distinct_episode_ids": "Distinct nonmissing NOAA episode identifiers within this county/year/category; not national disaster count.",
        "nri_material_eal_adjusted_usd": "Adjusted EALB + EALA, excluding population equivalence. Blank when either component is missing/not modeled.",
        "largest_event_material_loss_share": "Largest single record material loss / annual material loss; blank for incomplete or zero total.",
        "flash_flood_record_share": "Flash Flood records / retained records in this county/year/category.",
        "nri_price_source_month": "Price basis for NRI dollar fields, independent of release date; default 2024-12 for v1.20.",
        "nri_price_target_month": "Target month for CPI-adjusted NRI dollar fields.",
        "nri_cpi_factor": "CPI target month / CPI NRI price month; applied to USD fields only.",
        "nri_hazard_name": "HazardInfo hazard name selected by IFLD/CFLD prefix.",
        "nri_record_start": "HazardInfo Start; for probability models this is a dataset year, not a historical window.",
        "nri_record_end": "HazardInfo End; for probability models this is a dataset year, not a historical window.",
        "nri_record_years": "HazardInfo TotalYears; not the denominator for observed flood_final annual aggregation.",
        "nri_frequency_model": "HazardInfo FrequencyModel: Annualized Frequency or Probability.",
        "nri_record_description": "HazardInfo PeriodOfRecord description; retain the source qualification.",
        "source_year_differs_from_analysis_year": "1 if the source year differs from the EST start year; source year is preserved.",
    }
    metadata_sources = {
        "nri_price_source_month": "parameter --nri-price-month; FEMA NRI price-basis metadata",
        "nri_price_target_month": "parameter --base-month",
        "nri_cpi_factor": "CPIAUCSL target-month CPI / NRI source-month CPI",
        "nri_material_eal_adjusted_usd": "IFLD_EALB + IFLD_EALA, CPI-adjusted; blank for CFLD",
        "nri_hazard_name": "NRI_HazardInfo.csv: Hazard",
        "nri_record_start": "NRI_HazardInfo.csv: Start",
        "nri_record_end": "NRI_HazardInfo.csv: End",
        "nri_record_years": "NRI_HazardInfo.csv: TotalYears",
        "nri_frequency_model": "NRI_HazardInfo.csv: FrequencyModel",
        "nri_record_description": "NRI_HazardInfo.csv: PeriodOfRecord",
    }
    result = []
    for table, columns in schemas.items():
        for name in columns:
            source_field, definition, units = "", descriptions.get(name, ""), ""
            if name.startswith("nri_"):
                candidate = "NRI_ID" if name == "nri_id" else name[4:].removesuffix("_adjusted_usd").upper()
                if candidate in COMMON_FIELDS:
                    source_field = candidate
                elif candidate in HAZARD_SUFFIXES:
                    prefixes = ("IFLD",) if candidate in {"EXPA", "HLRA", "EALA", "ALRA"} else ("IFLD", "CFLD")
                    source_field = " / ".join(p + "_" + candidate for p in prefixes)
                else:
                    source_field = metadata_sources.get(name, "")
                item = nri_dictionary.get(candidate) or nri_dictionary.get("IFLD_" + candidate) or nri_dictionary.get("CFLD_" + candidate)
                if item:
                    definition = item["Field Alias"] + "; county snapshot, never sum across event rows or years."
                if name.endswith("_adjusted_usd"):
                    definition += " Converted using CPI(target)/CPI(NRI price month)."
                    units = "USD at target-month prices"
                elif candidate in COMMON_MONEY or candidate in MONEY_SUFFIXES:
                    units = "USD at NRI source-month prices"
                elif candidate == "POPULATION":
                    units = "people (2020 snapshot per source dictionary)"
                elif candidate in {"EXPP", "EALP"}:
                    units = "population / population-loss equivalent, not USD"
            elif name.endswith("_adjusted_usd"):
                definition = definition or "Sum of individually CPI-adjusted event amounts; blanks are not assumed zero."
                units = "USD at target-month prices"
            elif name in {"damage_property_usd", "damage_crops_usd"} or name.endswith("_original_usd"):
                definition = definition or "Original nominal flood loss; annual sums combine original dollars from event months."
                units = "nominal USD (mixed prices for annual sums)"
            elif "_to_nri_" in name or "_per_nri_population" in name or "_per_100k_nri_" in name:
                definition = "Exploratory ratio using a static NRI denominator; not contemporaneous exposure or causal effect. Blank for missing/zero denominator or incomplete numerator."
                if "building_eal" in name:
                    definition += " NOAA property damage is broader than NRI building damage; comparison is a proxy."
                units = "ratio" if name.endswith("ratio") else "USD/person" if name.endswith("usd") else "deaths/100000 people"
            if not definition:
                if name.endswith("_missing_records") or name.endswith("_known_records"):
                    definition = "Number of retained records with missing / known values for this metric."
                elif name in COUNTS:
                    definition = "Reported NOAA casualty count; annual known-value sum. Check corresponding missing-record count."
                elif name in {"county_fips", "state_fips"}:
                    definition = "Text identifier; preserve leading zeros."
                elif name.startswith("cpi_"):
                    definition = "Monthly CPI provenance; source and target price levels are kept explicitly."
                else:
                    definition = "Preserved flood_final field." if table == "flood_nri_events.csv" else "Grouping key or provenance for this table."
            result.append({"table": table, "field": name, "source_field": source_field,
                           "definition": definition, "units": units})
    return result


def combine(*, flood: Path, nri: Path, cleaned: Path, hazards: Path, dictionary: Path,
            cpi: Path, output_dir: Path, base_month="2026-08", nri_price_month="2024-12",
            missing_cpi="interpolate", overwrite=False) -> dict:
    inputs = [path.resolve() for path in (flood, nri, cleaned, hazards, dictionary, cpi)]
    output_dir = output_dir.resolve()
    outputs = [output_dir / name for name in OUTPUT_NAMES]
    if any(path in inputs for path in outputs):
        raise ValueError("Outputs must differ from inputs")
    if any(path.exists() for path in outputs) and not overwrite:
        raise ValueError("Output exists; use --overwrite to replace only the five generated files")
    if missing_cpi not in {"error", "interpolate"}:
        raise ValueError("Unknown missing CPI policy")
    cpi_values = read_cpi(cpi)
    base_value, _ = month_cpi(cpi_values, parse_month(base_month), interpolate=False)
    nri_value, _ = month_cpi(cpi_values, parse_month(nri_price_month), interpolate=False)
    full, cleaned_rows = read_counties(nri), read_counties(cleaned)
    first = next(iter(full.values()))
    needed = set(COMMON_FIELDS) | {"STATEFIPS"} | {"IFLD_" + s for s in HAZARD_SUFFIXES}
    needed |= {"CFLD_" + s for s in HAZARD_SUFFIXES if s not in {"EXPA", "HLRA", "EALA", "ALRA"}}
    if needed - first.keys():
        raise ValueError(f"Missing NRI fields: {sorted(needed - first.keys())}")
    if {row["NRI_VER"] for row in full.values()} != {"December 2025"}:
        raise ValueError("This mapping and default price basis support the December 2025 NRI release only")
    reconciliation = reconcile_cleaned(full, cleaned_rows)
    hazard_rows = {}
    for row in csv_rows(hazards, {"Prefix", "Hazard", "Start", "End", "TotalYears", "FrequencyModel", "PeriodOfRecord", "NRI_VER"}):
        if row["Prefix"] in hazard_rows:
            raise ValueError(f"Duplicate hazard prefix: {row['Prefix']}")
        hazard_rows[row["Prefix"]] = row
    if not {"IFLD", "CFLD"} <= hazard_rows.keys():
        raise ValueError("Missing inland/coastal hazard metadata")
    nri_dict = {row["Field Name"]: row for row in csv_rows(dictionary, {"Field Name", "Field Alias", "Version Date"})}
    for prefix in ("IFLD", "CFLD"):
        if hazard_rows[prefix]["NRI_VER"] != "December 2025":
            raise ValueError("Hazard metadata version disagrees with county table")
    for name in needed - {"NRI_VER"}:
        if name not in nri_dict or nri_dict[name]["Version Date"] != "December 2025":
            raise ValueError(f"Dictionary missing or wrong-version field: {name}")
    contexts = {(key, prefix): make_context(row, prefix, hazard_rows.get(prefix, {}), base_value,
                nri_value, nri_price_month, base_month)
                for key, row in full.items() for prefix in ("IFLD", "CFLD", "")}
    required_flood = {"event_id", "episode_id", "event_type", "year", "begin_time_est", "state_fips",
                      "county_fips", "area_type", "year_is_partial", "core_fields_valid",
                      "damage_property_usd", "damage_crops_usd", *COUNTS}
    with flood.open(encoding="utf-8-sig", newline="") as source:
        flood_fields = next(csv.reader(source))
    if required_flood - set(flood_fields) or len(flood_fields) != len(set(flood_fields)):
        raise ValueError("Invalid flood_final schema")
    if set(flood_fields) & (set(EVENT_DERIVED) | set(nri_columns())):
        raise ValueError("Input already contains adjusted/joined columns; use nominal flood_final.csv")
    event_fields = flood_fields + list(EVENT_DERIVED) + list(EVENT_CONTEXT_FIELDS)
    county_fields = ["county_fips", "nri_hazard_prefix", *nri_columns()]
    year_fields = annual_columns()
    schemas = dict(zip(OUTPUT_NAMES[:3], (event_fields, county_fields, year_fields)))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Stage each file in the destination so Windows renames inherit normal ACLs.
    staged = []
    try:
        for name in OUTPUT_NAMES:
            with tempfile.NamedTemporaryFile(prefix=".nri-flood-", suffix=Path(name).suffix,
                                             dir=output_dir, delete=False) as handle:
                staged.append(Path(handle.name))
        joins, type_counts, estimated_months, totals = Counter(), Counter(), Counter(), Counter()
        seen, groups, factors, years = set(), {}, {}, set()
        with staged[0].open("w", encoding="utf-8-sig", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=event_fields)
            writer.writeheader()
            for row in csv_rows(flood, required_flood):
                event_id = row["event_id"]
                if not re.fullmatch(r"[1-9]\d*", event_id) or event_id in seen:
                    raise ValueError(f"Invalid/duplicate event_id: {event_id}")
                seen.add(event_id)
                if row["core_fields_valid"] != "1" or row["year_is_partial"] not in {"0", "1"}:
                    raise ValueError(f"Invalid quality indicators in event {event_id}")
                event_type = row["event_type"]
                if event_type not in EVENT_HAZARDS:
                    raise ValueError(f"Unmapped event type: {event_type}")
                month = event_month(row["begin_time_est"])
                analysis_year = month.year
                source_year = int(row["year"])
                years.add(analysis_year)
                prefix = EVENT_HAZARDS[event_type]
                category = "inland" if prefix == "IFLD" else "coastal" if prefix == "CFLD" else "lakeshore"
                status = join_status(row, full)
                context = contexts.get((row["county_fips"], prefix), {}) if status == "matched_county" else {}
                # Join first, then CPI-adjust event money, then aggregate the adjusted rows.
                row.update({name: context.get(name, "") for name in EVENT_CONTEXT_FIELDS})
                if month not in factors:
                    factors[month] = month_cpi(cpi_values, month, interpolate=missing_cpi == "interpolate")
                source_value, estimated = factors[month]
                prop, crops = number(row["damage_property_usd"]), number(row["damage_crops_usd"])
                for basis in ("property", "crops"):
                    row["damage_" + basis + "_adjusted_usd"] = adjust_amount(row["damage_" + basis + "_usd"], base_value, source_value)
                adjusted_prop, adjusted_crops = number(row["damage_property_adjusted_usd"]), number(row["damage_crops_adjusted_usd"])
                row.update(
                    analysis_year=analysis_year, event_month=month.strftime("%Y-%m"), flood_category=category,
                    nri_hazard_prefix=prefix, nri_join_status=status,
                    nri_hazard_mapping_status="mapped" if prefix else "unmapped_lakeshore",
                    source_year_differs_from_analysis_year=int(source_year != analysis_year),
                    within_nri_hazard_record_period=record_period_flag(context, analysis_year),
                    cpi_source_month=month.strftime("%Y-%m"), cpi_source_value=str(source_value),
                    cpi_target_value=str(base_value), cpi_factor=str(base_value / source_value),
                    cpi_interpolated=int(estimated), cpi_base_month=base_month,
                    damage_material_original_usd=currency(prop + crops if prop is not None and crops is not None else None),
                    damage_material_adjusted_usd=currency(adjusted_prop + adjusted_crops if adjusted_prop is not None and adjusted_crops is not None else None),
                )
                writer.writerow(row)
                joins[status] += 1
                type_counts[event_type] += 1
                if estimated:
                    estimated_months[month.strftime("%Y-%m")] += 1
                for basis in LOSSES:
                    amount = row["damage_" + basis + "_adjusted_usd"]
                    if amount:
                        totals[basis + "_all"] += Decimal(amount)
                        totals[basis + ("_matched" if status == "matched_county" else "_unmatched")] += Decimal(amount)
                if status == "matched_county":
                    group = groups.setdefault((row["county_fips"], analysis_year, category), AnnualGroup())
                    group.add(row)
        if not seen:
            raise ValueError("Empty flood input")
        write_csv(staged[1], county_fields,
                  ({"county_fips": key, "nri_hazard_prefix": prefix, **contexts[key, prefix]}
                   for key in sorted(full) for prefix in ("IFLD", "CFLD")))
        annual_totals = Counter()
        def annual_rows():
            for (key, year, category), group in sorted(groups.items()):
                prefix = "IFLD" if category == "inland" else "CFLD" if category == "coastal" else ""
                context = contexts[key, prefix]
                metrics = group.result(context)
                for basis in LOSSES:
                    annual_totals[basis] += group.sums[basis + "_adjusted"]
                yield {"county_fips": key, "analysis_year": year, "flood_category": category,
                       "nri_hazard_prefix": prefix, "within_nri_hazard_record_period": record_period_flag(context, year),
                       "cpi_base_month": base_month, **metrics, **context}
        write_csv(staged[2], year_fields, annual_rows())
        if sum(group.events for group in groups.values()) != joins["matched_county"]:
            raise ValueError("County-year record conservation failed")
        for basis in LOSSES:
            if annual_totals[basis] != totals[basis + "_matched"]:
                raise ValueError(f"County-year loss conservation failed: {basis}")
        write_csv(staged[3], ["table", "field", "source_field", "definition", "units"],
                  field_dictionary(schemas, nri_dict))
        metadata = {
            "inputs": [str(path) for path in inputs],
            "parameters": {"base_month": base_month, "base_cpi": str(base_value), "nri_price_month": nri_price_month,
                           "nri_price_cpi": str(nri_value), "nri_price_factor": str(base_value / nri_value),
                           "missing_cpi": missing_cpi, "nri_price_basis_source": PRICE_SOURCE},
            "output_grains": {OUTPUT_NAMES[0]: "one row per input event, including unmatched records",
                              OUTPUT_NAMES[1]: "one row per NRI county and IFLD/CFLD hazard",
                              OUTPUT_NAMES[2]: "one row per matched county, EST start year, flood category with retained records"},
            "row_counts": {OUTPUT_NAMES[0]: len(seen), OUTPUT_NAMES[1]: len(full) * 2,
                           OUTPUT_NAMES[2]: len(groups)},
            "analysis_year_range": [min(years), max(years)], "join_status_counts": dict(sorted(joins.items())),
            "event_type_counts": dict(sorted(type_counts.items())),
            "cpi_interpolated_months": {month: {"event_records": count, "cpi": str(factors[parse_month(month)][0])}
                                        for month, count in sorted(estimated_months.items())},
            "adjusted_loss_controls_usd": {name: currency(value) for name, value in sorted(totals.items())},
            "cleaned_reconciliation": reconciliation,
            "methods_and_limits": [
                "Input is nominal flood_final.csv; source damage fields are preserved and adjusted columns are separate.",
                "Flood CPI uses fixed-EST event start month, consistent with adjust_flood_cpi.py. Multimonth losses are not split without observations.",
                "NRI USD fields use the declared December 2024 price basis; release date December 2025 is not the price month.",
                "County and hazard joins are many-to-one; NRI snapshots must never be summed over event rows or years.",
                "Event rows include selected predictors; join (county_fips,nri_hazard_prefix) to nri_flood_counties for complete hazard context. County-year rows already include it.",
                "Forecast zones and historical county codes are retained as unmatched; no invented allocation. See join status.",
                "Lakeshore Flood has no assumed IFLD/CFLD equivalence; county context is attached but hazard metrics stay blank.",
                "No absent county-years are manufactured as zero. Event counts describe the damage-screened flood_final sample.",
                "Source cleaning removed both-loss-missing records and filled one missing loss component with zero; this table cannot reverse that imputation.",
                "Source casualty missingness is retained. Annual known-value sums have missing-record counts; incomplete ratios remain blank.",
                "Missing NRI numbers are not zero. Blank coastal agriculture fields mean not modeled, not zero loss.",
                "POPULATION is a 2020 snapshot per NRIDataDictionary; asset and social metrics are static snapshots, not historical county-year measures.",
                "NOAA property damage and NRI building loss differ in scope; loss/EAL ratios are descriptive proxies, not model validation or causal effects.",
                "Only NRI annualized-frequency hazards receive a historical-period flag. Probability-model dates are not observational windows.",
                "Annual physical losses exclude monetized population loss; population equivalent USD remains a separate NRI metric.",
            ],
        }
        staged[4].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Publish only after all input rows and aggregate controls succeed.
        if not overwrite and any(path.exists() for path in outputs):
            raise ValueError("Output appeared during processing; refusing to overwrite")
        for temporary, destination in zip(staged, outputs):
            if overwrite:
                temporary.replace(destination)
            else:
                temporary.rename(destination)
        return metadata
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flood", type=Path, default=ROOT / "data/processed_data/flood_final.csv")
    parser.add_argument("--nri", type=Path, default=ROOT / "NRI_Table_Counties.csv")
    parser.add_argument("--cleaned", type=Path, default=ROOT / "NRI_Table_Counties_Cleaned.csv")
    parser.add_argument("--hazards", type=Path, default=ROOT / "NRI_HazardInfo.csv")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "NRIDataDictionary.csv")
    parser.add_argument("--cpi", type=Path, default=ROOT / "CPIAUCSL.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/analyzed_data")
    parser.add_argument("--base-month", default="2026-08")
    parser.add_argument("--nri-price-month", default="2024-12", help="NRI dollar price basis, not publication date")
    parser.add_argument("--missing-cpi", choices=("error", "interpolate"), default="interpolate")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        summary = combine(**vars(args))
    except (ValueError, OSError, csv.Error, InvalidOperation, OverflowError) as error:
        print(f"Merge failed: {error}", file=sys.stderr)
        return 1
    print(f"Price basis: {summary['parameters']['base_month']}; CPI={summary['parameters']['base_cpi']}")
    for name, count in summary["row_counts"].items():
        print(f"{name}: {count:,} rows")
    print("County joins: " + json.dumps(summary["join_status_counts"], ensure_ascii=False))
    print("CPI interpolation: " + json.dumps(summary["cpi_interpolated_months"], ensure_ascii=False))
    print(f"Output directory: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
