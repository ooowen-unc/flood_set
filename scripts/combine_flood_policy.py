from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from statistics import median

try:
    from .adjust_flood_cpi import read_cpi, month_cpi
    from .combine_flood_nri import csv_rows, read_counties
except ImportError:
    from adjust_flood_cpi import read_cpi, month_cpi
    from combine_flood_nri import csv_rows, read_counties

ROOT = Path(__file__).resolve().parents[1]
CENT = Decimal("0.01")
FLOOD_TYPES = {
    "Flash Flood": "flash_flood", "Flood": "inland_flood",
    "Coastal Flood": "coastal_flood", "Lakeshore Flood": "lakeshore_flood",
}
CASUALTIES = ("injuries_direct", "injuries_indirect", "deaths_direct", "deaths_indirect")
PA_CATEGORIES = {"A": "debris", "B": "emergency", "C": "roads_bridges",
                 "D": "water_control", "E": "buildings", "F": "utilities",
                 "G": "parks_other", "Z": "administration", "I": "building_codes"}
POSSIBLE_FLOOD_INCIDENTS = {"Hurricane", "Tropical Storm", "Severe Storm(s)",
                          "Severe Storm", "Coastal Storm", "Typhoon", "Dam/Levee Break",
                          "Severe Storms, Straight-line Winds, Tornadoes, and Flooding"}
AWARDED = {"Approved", "Awarded", "Obligated", "Closed", "Completed"}
CLOSED = {"Closed", "Completed"}
FLOOD_PROGRAMS = {"FMA", "SRL", "RFC"}
PROGRAMS = ("HMGP", "FMA", "PDM", "LPDM", "BRIC", "SRL", "RFC")
MEASURES = ("acquisition", "relocation", "elevation", "floodproofing",
            "drainage", "flood_control", "restoration", "infrastructure")
NRI_COMMON = ("NRI_ID", "STATE", "STATEABBRV", "COUNTY", "COUNTYTYPE", "NRI_VER",
              "POPULATION", "BUILDVALUE", "AGRIVALUE", "AREA", "SOVI_SCORE",
              "SOVI_RATNG", "RESL_SCORE", "RESL_RATNG", "RESL_VALUE", "CRF_VALUE")
NRI_SUFFIXES = ("EVNTS", "AFREQ", "EXPB", "EXPP", "EXPA", "HLRB", "HLRP", "HLRA",
                "EALB", "EALP", "EALPE", "EALA", "EALT", "EALS", "EALR",
                "RISKS", "RISKR", "ALRB", "ALRA", "ALR_NPCTL")


def numeric(text: str, *, nonnegative: bool = False) -> Decimal | None:
    if not text.strip():
        return None
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid number: {text!r}") from exc
    if not value.is_finite() or (nonnegative and value < 0):
        raise ValueError(f"Invalid finite/nonnegative number: {text!r}")
    return value


def annual_year(text: str, maximum: int, minimum: int = 1989) -> int | None:
    """Use the recorded calendar year only; never repair implausible years."""
    match = re.match(r"^(\d{4})(?:-|$)", text.strip())
    if not match:
        return None
    year = int(match[1])
    return year if minimum <= year <= maximum else None


def county_key(state: str, county: str) -> str | None:
    if not re.fullmatch(r"\d{1,2}", state.strip()) or not re.fullmatch(r"\d{1,3}", county.strip()):
        return None
    if int(state) == 0 or int(county) == 0:
        return None
    return state.strip().zfill(2) + county.strip().zfill(3)


def county_name(text: str) -> str:
    text = re.sub(r"\s+(COUNTY|PARISH|BOROUGH|CENSUS AREA|MUNICIPALITY)$", "", text.strip().upper())
    return re.sub(r"[^A-Z0-9]", "", text)


def measures(text: str) -> set[str]:
    result = set()
    for token in text.split(";"):
        match = re.match(r"^([0-9]+\.[0-9]+)(?:A)?\s*:", token.strip())
        if not match:
            continue
        code = match[1]
        if code in {f"200.{i}" for i in range(1, 5)}:
            result.add("acquisition")
        elif code in {f"201.{i}" for i in range(1, 5)}:
            result.add("relocation")
        elif code in {f"202.{i}" for i in range(1, 5)}:
            result.add("elevation")
        elif code in {f"{p}.{i}" for p in (203, 204) for i in range(1, 5)}:
            result.add("floodproofing")
        elif code in {"403.1", "403.2", "403.3", "403.4", "403.5", "403.8"}:
            result.add("drainage")
        elif code in {"404.1", "405.1", "500.1", "500.2", "500.3"}:
            result.add("flood_control")
        elif code in {"303.2", "303.3"}:
            result.add("restoration")
        elif code in {"402.2", "402.4"}:
            result.add("infrastructure")
    return result


@dataclass
class Sum:
    total: Decimal = Decimal(0)
    known: int = 0
    missing: int = 0

    def add(self, value: Decimal | None):
        if value is None:
            self.missing += 1
        else:
            self.total += value
            self.known += 1

    def display(self) -> str:
        return format(self.total.quantize(CENT, rounding=ROUND_HALF_UP), "f") if self.known else ""


@dataclass
class Bucket:
    counts: Counter = field(default_factory=Counter)
    sums: dict[str, Sum] = field(default_factory=dict)
    sets: dict[str, set] = field(default_factory=lambda: defaultdict(set))
    values: dict[str, list[Decimal]] = field(default_factory=lambda: defaultdict(list))

    def add(self, name: str, value: Decimal | None):
        self.sums.setdefault(name, Sum()).add(value)


class Schema:
    def __init__(self):
        self.columns: dict[str, dict] = {}
        self.money: set[str] = set()
        self.counts: set[str] = set()

    def define(self, name: str, source: str, definition: str, units: str = "", role: str = "context"):
        if name in self.columns:
            raise ValueError(f"Duplicate output field: {name}")
        self.columns[name] = dict(field=name, source=source, definition=definition, units=units, role=role)

    def count(self, name: str, source: str, definition: str, role: str = "context"):
        self.define(name, source, definition, "records", role)
        self.counts.add(name)

    def amount(self, name: str, source: str, definition: str, units: str = "nominal USD", role: str = "snapshot_response"):
        self.define(name, source, definition + " Blank if no known contributing value; signed adjustments retained.", units, role)
        self.count(name + "_known_records", source, "Contributing records with a known value.", role)
        self.count(name + "_missing_records", source, "Contributing records whose value is blank.", role)
        self.money.add(name)


def build_schema(base_year: int) -> Schema:
    s = Schema()
    for name, meaning in {
        "county_fips": "Five-character source county FIPS; historical codes retained.",
        "analysis_year": "Calendar year, one row per county with retained flood observations.",
        "county_name": "NRI county name if matched; otherwise observed NOAA area name.",
        "nri_match_status": "matched or unmatched_county_fips; unmatched counties are retained.",
        "cpi_base_year": "Annual CPI price target year; applies only to adjusted flood loss fields.",
        "cpi_source_interpolated_observations": "Missing monthly CPI observations filled before calculating source annual mean.",
        "cpi_base_interpolated_observations": "Missing CPI observations filled before calculating target annual mean.",
    }.items():
        s.define(name, "derived/local CSVs", meaning, role="key" if name in {"county_fips", "analysis_year"} else "context")
    for name, meaning in {
        "flood_records": "Recorded, damage-screened flood rows in this county/year; not all flood occurrence.",
        "flood_distinct_episodes": "Distinct nonblank NOAA episode IDs within this county/year.",
        "flood_missing_episode_records": "Records with no episode ID.",
        "flood_location_records": "Records carrying reported location points; not inundation polygons.",
        "flood_partial_source_year_records": "Records flagged partial by the source cleaning.",
    }.items():
        s.count(name, "flood_final.csv", meaning, "observed_impact")
    for slug in FLOOD_TYPES.values():
        s.count(f"flood_{slug}_records", "flood_final.csv:event_type", "Recorded event count for this flood subtype.", "observed_impact")
        s.amount(f"flood_{slug}_material_loss_nominal_usd", "flood_final.csv", "Property plus crop loss for this subtype.", role="observed_impact")
    for kind in ("property", "crops", "material"):
        s.amount(f"flood_{kind}_loss_nominal_usd", "flood_final.csv", f"Known-value sum of {kind} damage.", role="observed_impact")
        s.amount(f"flood_{kind}_loss_adjusted_usd", "flood_final.csv+CPIAUCSL.csv", f"{kind} damage adjusted with annual CPI; each event rounded to cents.", f"{base_year} annual-CPI USD", "observed_impact")
    for name in CASUALTIES:
        s.amount("flood_" + name, "flood_final.csv", "Known-value sum of reported casualties; inspect missing count.", "people", "observed_impact")
    s.define("flood_mean_duration_hours", "flood_final.csv", "Mean of known event durations.", "hours", "observed_impact")
    s.define("flood_max_duration_hours", "flood_final.csv", "Maximum of known event durations.", "hours", "observed_impact")
    s.count("flood_duration_missing_records", "flood_final.csv", "Missing event duration.", "observed_impact")
    s.define("flood_cause_counts", "flood_final.csv", "JSON counts of nonblank flood_cause, with missing cause separately counted.", role="observed_impact")
    s.count("flood_cause_missing_records", "flood_final.csv", "Blank reported flood cause.", "observed_impact")
    for name in NRI_COMMON:
        s.define("nri_" + name.lower(), "NRI_Table_Counties.csv:" + name,
                 "Unmodified NRI static snapshot field; not a historical annual observation.", role="static_snapshot")
    for prefix in ("IFLD", "CFLD"):
        for suffix in NRI_SUFFIXES:
            s.define("nri_" + prefix.lower() + "_" + suffix.lower(), "NRI_Table_Counties.csv:" + prefix + "_" + suffix,
                     "Unmodified hazard-specific static NRI field; blanks are not zeros and money retains NRI source price basis.", role="static_snapshot")
    for name in ("pa_records", "pa_eligible_records", "pa_noneligible_records", "pa_distinct_disasters",
                 "pa_flood_incident_eligible_records", "pa_possible_flood_incident_eligible_records"):
        s.count(name, "PublicAssistanceFundedProjectsDetails.csv", "County of applicant; calendar declaration-year cohort. No attribution to a particular NOAA event.", "snapshot_response")
    for source, label in (("projectAmount", "project_amount"), ("federalShareObligated", "federal_share_obligated"),
                          ("totalObligated", "total_obligated"), ("mitigationAmount", "mitigation_amount")):
        s.amount(f"pa_eligible_{label}_nominal_usd", "PA:" + source,
                 "Eligible projects assigned to declaration year; cumulative value at source snapshot, not annual payments.")
    for kind in ("flood_incident", "possible_flood_incident"):
        s.amount(f"pa_{kind}_federal_share_obligated_nominal_usd", "PA:incidentType,federalShareObligated",
                 "Exact Flood incident type or separately designated flood-capable incident types; not flood-exclusive expenditure.")
    for label in list(PA_CATEGORIES.values()) + ["other_category"]:
        s.count(f"pa_eligible_{label}_records", "PA:damageCategoryCode", "Eligible projects by work category, assigned once to declaration year.", "snapshot_response")
        s.amount(f"pa_eligible_{label}_federal_share_nominal_usd", "PA:federalShareObligated", "Cumulative eligible federal share for this work category.")
    s.define("pa_status_counts", "PA:projectStatus", "JSON distribution of snapshot eligibility status.", role="snapshot_response")
    s.define("pa_process_step_counts", "PA:projectProcessStep", "JSON distribution of snapshot process stage.", role="snapshot_response")
    s.count("pa_negative_federal_share_records", "PA:federalShareObligated", "Eligible records with a negative adjustment.", "snapshot_response")
    for name in ("hma_fy_records", "hma_fy_awarded_records", "hma_fy_pending_records", "hma_fy_closed_records",
                 "hma_fy_not_awarded_records", "hma_fy_flood_related_records", "hma_fy_flood_awarded_records",
                 "hma_fy_multicounty_coverage_records", "hma_fy_duplicate_business_identifier_records",
                 "hma_fy_invalid_year_records", "hma_fy_invalid_chronology_records",
                 "hma_fy_invalid_cost_share_records", "hma_fy_bcr_zero_records",
                 "hma_fy_bcr_missing_records", "hma_fy_flood_with_pa_disaster_records",
                 "hma_fy_flood_with_pa_same_county_records"):
        s.count(name, "HazardMitigationAssistanceProjects.csv", "Snapshot records in program fiscal-year cohort; coverage counts may appear in multiple counties.", "snapshot_response")
    for program in PROGRAMS:
        s.count("hma_fy_" + program.lower() + "_records", "HMA:programArea", "Program fiscal-year cohort records in this county.", "snapshot_response")
    for measure in MEASURES:
        s.count(f"hma_fy_flood_awarded_{measure}_records", "HMA:projectType,status,programFy",
                "Awarded flood-related fiscal-year cohort records with this activity; multiple activities and county coverage are non-additive.", "snapshot_response")
        s.count(f"hma_flood_first_approval_year_{measure}_records", "HMA:projectType,dateInitiallyApproved",
                "Flood-related awarded records with this activity first approved in analysis_year; no substitute for a missing first approval year.", "snapshot_response")
    for source, label in (("projectAmount", "project_cost"), ("federalShareObligated", "federal_share_obligated"),
                          ("subrecipientAdminCostAmt", "subrecipient_admin"), ("srmcObligatedAmt", "subrecipient_management"),
                          ("recipientAdminCostAmt", "recipient_admin")):
        s.amount("hma_fy_flood_awarded_singlecounty_" + label + "_nominal_usd", "HMA:" + source,
                 "Flood-related awarded records in fiscal-year cohort; only single-county scope, no spatial cost allocation.")
    s.define("hma_fy_flood_awarded_positive_bcr_median", "HMA:benefitCostRatio", "Median of positive finite BCRs; zero, missing and outliers are not evidence of realized effectiveness.", role="snapshot_response")
    s.define("hma_fy_status_counts", "HMA:status", "JSON distribution of fiscal-year cohort snapshot statuses.", role="snapshot_response")
    for stage in ("first_approval", "latest_approval", "initial_obligation", "closed"):
        s.count(f"hma_flood_{stage}_year_records", "HMA:milestone dates", "Flood-related awarded records whose recorded milestone YEAR equals analysis_year; stages must not be added together.", "snapshot_response")
    s.amount("hma_flood_initial_obligation_year_singlecounty_nominal_usd", "HMA:initialObligationAmount",
             "Initial obligation amount in its recorded calendar year, single-county scope only; not all obligations or payments.")
    s.count("hma_flood_closed_before_year_records", "HMA:dateClosed,status", "Flood-related closed records with closure YEAR strictly before analysis_year; includes pre-1999 history.", "prior_mitigation_context")
    s.count("hma_physical_flood_closed_before_year_records", "HMA:projectType,dateClosed", "Prior-year closed records with an explicit physical flood measure; includes multiple activity projects.", "prior_mitigation_context")
    for measure in MEASURES:
        s.count(f"hma_{measure}_closed_before_year_records", "HMA:projectType,dateClosed", "Prior closure-year project coverage with this activity; multiple activities are non-additive.", "prior_mitigation_context")
    for name in ("pa_county_year_record_observed", "hma_fiscal_cohort_record_observed"):
        s.define(name, "derived", "1 if matching source records observed; 0 means no matching records, not no government action.", "indicator", "observation_flag")
    return s


def annual_cpi(path: Path, years: set[int]) -> tuple[dict[int, Decimal], dict[int, int], dict[int, Decimal]]:
    monthly = read_cpi(path)
    averages, interpolations, totals = {}, {}, {}
    for year in sorted(years):
        samples, filled = [], 0
        for month in range(1, 13):
            value, estimated = month_cpi(monthly, date(year, month, 1), interpolate=True)
            samples.append(value)
            filled += int(estimated)
        totals[year] = sum(samples)
        averages[year] = totals[year] / Decimal(12)
        interpolations[year] = filled
    return averages, interpolations, totals


def annual_loss(value: Decimal | None, target_total: Decimal, source_total: Decimal) -> Decimal | None:
    # Both annual means have the same denominator (12). Cancel it before
    # division, and multiply first, to retain exact half-cent rounding boundaries.
    return None if value is None else (value * target_total / source_total).quantize(CENT, rounding=ROUND_HALF_UP)


def hma_locations(row: dict, aliases: dict) -> tuple[list[str], str]:
    primary = county_key(row["stateNumberCode"], row["countyCode"])
    if not row["county"].strip() or county_name(row["county"]) == "STATEWIDE":
        primary = None
    names = [n.strip() for n in row["projectCounties"].split(";") if n.strip()]
    if not names:
        return ([primary], "primary_only") if primary else ([], "missing_location")
    if any(county_name(n) == "STATEWIDE" for n in names):
        return [], "statewide"
    found = set()
    for name in names:
        matches = aliases.get((row["stateNumberCode"].zfill(2), county_name(name)), set())
        if len(matches) != 1:
            return [], "unresolved_or_ambiguous_county_name"
        found.update(matches)
    if primary and primary not in found:
        return [], "primary_and_coverage_conflict"
    return sorted(found), "single_county" if len(found) == 1 else "multiple_counties"


def merge_annual(flood_path: Path, pa_path: Path, hma_path: Path, nri_path: Path,
                 cpi_path: Path, output_dir: Path, *, start_year: int = 1999,
                 end_year: int = 2025, base_year: int = 2025, overwrite: bool = False) -> dict:
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year")
    inputs = [Path(p).resolve() for p in (flood_path, pa_path, hma_path, nri_path, cpi_path)]
    output_dir = Path(output_dir).resolve()
    stem = f"flood_policy_county_year_{start_year}_{end_year}"
    outputs = [output_dir / (stem + ".csv"), output_dir / (stem + "_dictionary.csv"), output_dir / (stem + "_metadata.json")]
    for path in outputs:
        if path in inputs or (path.exists() and not overwrite):
            raise ValueError(f"Refusing to overwrite input/existing output: {path}; use --overwrite for outputs")
    schema = build_schema(base_year)
    cpi, cpi_fill, cpi_totals = annual_cpi(inputs[4], set(range(start_year, end_year + 1)) | {base_year})
    nri = read_counties(inputs[3])
    aliases = defaultdict(set)
    for cf, row in nri.items():
        for name in (row.get("COUNTY", ""), row.get("COUNTY", "") + " " + row.get("COUNTYTYPE", "")):
            if name.strip():
                aliases[(cf[:2], county_name(name))].add(cf)
    buckets = defaultdict(Bucket)
    controls = defaultdict(Counter)
    omissions = defaultdict(Sum)
    source_counts = Counter()
    seen = set()
    for row in csv_rows(inputs[0], {"event_id", "begin_time_est", "county_fips", "area_type", "event_type", "damage_property_usd", "damage_crops_usd"}):
        source_counts["flood"] += 1
        if not row["event_id"] or row["event_id"] in seen:
            raise ValueError("Missing/duplicate NOAA event_id")
        seen.add(row["event_id"])
        year = annual_year(row["begin_time_est"], end_year, start_year)
        if year is None:
            controls["flood"]["outside_output_years"] += 1
            continue
        prop = numeric(row["damage_property_usd"], nonnegative=True)
        crop = numeric(row["damage_crops_usd"], nonnegative=True)
        material = prop + crop if prop is not None and crop is not None else None
        cf = row["county_fips"]
        reason = "forecast_zone" if row["area_type"] == "Z" else "missing_or_invalid_county"
        if row["area_type"] != "C" or not re.fullmatch(r"\d{5}", cf) or cf.endswith("000"):
            controls["flood"][reason] += 1
            omissions[reason].add(material)
            continue
        controls["flood"]["included_records"] += 1
        aliases[(cf[:2], county_name(row.get("area_name", "")))].add(cf)
        b = buckets[(cf, year)]
        b.counts["flood_records"] += 1
        if row.get("episode_id"):
            b.sets["flood_distinct_episodes"].add(row["episode_id"])
        else:
            b.counts["flood_missing_episode_records"] += 1
        b.sets["area_names"].add(row.get("area_name", ""))
        b.counts["flood_location_records"] += bool(row.get("location_points"))
        b.counts["flood_partial_source_year_records"] += row.get("year_is_partial") == "1"
        subtype = FLOOD_TYPES.get(row["event_type"])
        if subtype is None:
            raise ValueError(f"Unsupported NOAA flood type: {row['event_type']}")
        b.counts[f"flood_{subtype}_records"] += 1
        b.add(f"flood_{subtype}_material_loss_nominal_usd", material)
        for kind, value in (("property", prop), ("crops", crop), ("material", material)):
            b.add(f"flood_{kind}_loss_nominal_usd", value)
            b.add(f"flood_{kind}_loss_adjusted_usd", annual_loss(value, cpi_totals[base_year], cpi_totals[year]))
        for name in CASUALTIES:
            b.add("flood_" + name, numeric(row.get(name, ""), nonnegative=True))
        duration = numeric(row.get("duration_hours", ""), nonnegative=True)
        if duration is None:
            b.counts["flood_duration_missing_records"] += 1
        else:
            b.values["duration"].append(duration)
        cause = row.get("flood_cause", "")
        b.counts["cause:" + cause] += 1
    print(f"Flood: {controls['flood']['included_records']:,} rows -> {len(buckets):,} county-years", flush=True)

    hma = list(csv_rows(inputs[2], {"id", "projectIdentifier", "programFy", "countyCode", "projectCounties", "projectType", "status"}))
    source_counts["hma"] = len(hma)
    identifiers = Counter(row["projectIdentifier"] for row in hma)
    controls["hma"]["duplicate_business_identifiers"] = sum(count > 1 for count in identifiers.values())
    h_ids = set()
    for row in hma:
        if not row["id"] or row["id"] in h_ids:
            raise ValueError("Missing/duplicate HMA record id")
        h_ids.add(row["id"])
        cf = county_key(row["stateNumberCode"], row["countyCode"])
        if cf and row["county"].strip() and county_name(row["county"]) != "STATEWIDE":
            aliases[(cf[:2], county_name(row["county"]))].add(cf)

    pa_declarations = defaultdict(set)
    pa_county_declarations = set()
    seen.clear()
    max_source_year = end_year
    pa_fields = {"projectAmount": "project_amount", "federalShareObligated": "federal_share_obligated",
                 "totalObligated": "total_obligated", "mitigationAmount": "mitigation_amount"}
    for row in csv_rows(inputs[1], {"gmProjectId", "disasterNumber", "declarationDate", "countyCode", "projectStatus", "federalShareObligated"}):
        source_counts["pa"] += 1
        pid = row["gmProjectId"]
        if not pid or pid in seen:
            raise ValueError("Missing/duplicate PA gmProjectId")
        seen.add(pid)
        refresh_year = annual_year(row.get("lastRefresh", ""), 2100)
        if refresh_year:
            max_source_year = max(max_source_year, refresh_year)
        year = annual_year(row["declarationDate"], 2100)
        state = row["stateNumberCode"].strip().zfill(2)
        if year:
            pa_declarations[(row["disasterNumber"], state)].add(year)
        cf = county_key(row["stateNumberCode"], row["countyCode"])
        location_ok = cf and row["county"].strip() and county_name(row["county"]) != "STATEWIDE"
        if location_ok:
            pa_county_declarations.add((row["disasterNumber"], cf))
        if year is None or not start_year <= year <= end_year:
            controls["pa"]["outside_or_invalid_declaration_year"] += 1
            continue
        value = numeric(row["federalShareObligated"])
        if not location_ok:
            controls["pa"]["unallocated_location_records"] += 1
            omissions["pa_unallocated_location_federal_share"].add(value)
            continue
        b = buckets.get((cf, year))
        if b is None:
            controls["pa"]["no_flood_county_year_records"] += 1
            omissions["pa_no_flood_county_year_federal_share"].add(value)
            continue
        controls["pa"]["included_records"] += 1
        b.counts["pa_records"] += 1
        b.sets["pa_distinct_disasters"].add(row["disasterNumber"])
        b.counts["pa_status:" + row["projectStatus"]] += 1
        b.counts["pa_step:" + row["projectProcessStep"]] += 1
        if row["projectStatus"] != "Eligible":
            b.counts["pa_noneligible_records"] += 1
            continue
        b.counts["pa_eligible_records"] += 1
        b.counts["pa_negative_federal_share_records"] += value is not None and value < 0
        for source, label in pa_fields.items():
            b.add(f"pa_eligible_{label}_nominal_usd", numeric(row[source]))
        category = PA_CATEGORIES.get(row["damageCategoryCode"], "other_category")
        b.counts[f"pa_eligible_{category}_records"] += 1
        b.add(f"pa_eligible_{category}_federal_share_nominal_usd", value)
        kind = "flood_incident" if row["incidentType"] == "Flood" else "possible_flood_incident" if row["incidentType"] in POSSIBLE_FLOOD_INCIDENTS else None
        if kind:
            b.counts[f"pa_{kind}_eligible_records"] += 1
            b.add(f"pa_{kind}_federal_share_obligated_nominal_usd", value)
    print(f"PA: {source_counts['pa']:,} rows; {controls['pa']['included_records']:,} annual context matches", flush=True)

    closed_deltas = defaultdict(Counter)
    hma_fields = {"projectAmount": "project_cost", "federalShareObligated": "federal_share_obligated",
                  "subrecipientAdminCostAmt": "subrecipient_admin", "srmcObligatedAmt": "subrecipient_management",
                  "recipientAdminCostAmt": "recipient_admin"}
    for row in hma:
        locations, location_status = hma_locations(row, aliases)
        controls["hma_locations"][location_status] += 1
        fy = annual_year(row["programFy"], max_source_year)
        actions = measures(row["projectType"])
        flood_related = bool(actions) or row["programArea"] in FLOOD_PROGRAMS
        awarded = row["status"] in AWARDED
        years = {name: annual_year(row.get(name, ""), max_source_year) for name in
                 ("dateInitiallyApproved", "dateApproved", "dateClosed", "initialObligationDate")}
        invalid_year = bool(row["programFy"].strip() and fy is None) or any(
            row.get(name, "").strip() and year is None for name, year in years.items())
        close = years["dateClosed"]
        invalid_chronology = close is not None and any(year is not None and year > close for name, year in years.items() if name != "dateClosed")
        if years["dateInitiallyApproved"] is not None and years["dateApproved"] is not None:
            invalid_chronology |= years["dateInitiallyApproved"] > years["dateApproved"]
        controls["hma"]["invalid_year_records"] += bool(invalid_year)
        controls["hma"]["invalid_chronology_records"] += bool(invalid_chronology)
        if not locations:
            if fy is not None and start_year <= fy <= end_year and flood_related and awarded:
                omissions["hma_unallocated_location_federal_share"].add(numeric(row["federalShareObligated"]))
            continue
        single = len(locations) == 1
        if not single and fy is not None and start_year <= fy <= end_year and flood_related and awarded:
            omissions["hma_multicounty_unallocated_federal_share"].add(numeric(row["federalShareObligated"]))
        for cf in locations:
            b = buckets.get((cf, fy))
            if b is not None:
                controls["hma"]["fiscal_cohort_county_year_coverage_links"] += 1
                b.counts["hma_fy_records"] += 1
                b.counts["hma_fy_awarded_records"] += awarded
                b.counts["hma_fy_pending_records"] += row["status"] in {"Pending", "Revision Requested"}
                b.counts["hma_fy_closed_records"] += row["status"] in CLOSED
                b.counts["hma_fy_not_awarded_records"] += not awarded
                b.counts["hma_fy_flood_related_records"] += flood_related
                b.counts["hma_fy_flood_awarded_records"] += flood_related and awarded
                b.counts["hma_fy_multicounty_coverage_records"] += not single
                b.counts["hma_fy_duplicate_business_identifier_records"] += identifiers[row["projectIdentifier"]] > 1
                b.counts["hma_fy_invalid_year_records"] += bool(invalid_year)
                b.counts["hma_fy_invalid_chronology_records"] += bool(invalid_chronology)
                share = numeric(row["costSharePercentage"])
                b.counts["hma_fy_invalid_cost_share_records"] += share is not None and not 0 <= share <= 1
                bcr = numeric(row["benefitCostRatio"])
                b.counts["hma_fy_bcr_zero_records"] += bcr == 0
                b.counts["hma_fy_bcr_missing_records"] += bcr is None
                b.counts["hma_status:" + row["status"]] += 1
                if row["programArea"] in PROGRAMS:
                    b.counts["hma_fy_" + row["programArea"].lower() + "_records"] += 1
                if flood_related:
                    b.counts["hma_fy_flood_with_pa_disaster_records"] += (row["disasterNumber"], cf[:2]) in pa_declarations
                    b.counts["hma_fy_flood_with_pa_same_county_records"] += (row["disasterNumber"], cf) in pa_county_declarations
                if flood_related and awarded:
                    for action in actions:
                        b.counts[f"hma_fy_flood_awarded_{action}_records"] += 1
                    if bcr is not None and bcr > 0:
                        b.values["bcr"].append(bcr)
                    if single:
                        for source, label in hma_fields.items():
                            b.add("hma_fy_flood_awarded_singlecounty_" + label + "_nominal_usd", numeric(row[source]))
            if not flood_related or not awarded:
                continue
            first_approval = years["dateInitiallyApproved"]
            # Keep amendment years separate; never infer a missing first approval.
            for stage, year in (("first_approval", first_approval), ("latest_approval", years["dateApproved"]),
                                ("initial_obligation", years["initialObligationDate"]), ("closed", close)):
                target = buckets.get((cf, year))
                if target is not None and not invalid_year and not invalid_chronology and (stage != "closed" or row["status"] in CLOSED):
                    target.counts[f"hma_flood_{stage}_year_records"] += 1
                    if stage == "first_approval":
                        for action in actions:
                            target.counts[f"hma_flood_first_approval_year_{action}_records"] += 1
                    if stage == "initial_obligation" and single:
                        target.add("hma_flood_initial_obligation_year_singlecounty_nominal_usd", numeric(row["initialObligationAmount"]))
            if row["status"] in CLOSED and close is not None and not invalid_year and not invalid_chronology:
                delta = closed_deltas[(cf, close)]
                delta["hma_flood_closed_before_year_records"] += 1
                delta["hma_physical_flood_closed_before_year_records"] += bool(actions)
                for action in actions:
                    delta[f"hma_{action}_closed_before_year_records"] += 1
    print(f"HMA: {source_counts['hma']:,} rows; annual milestones and prior-year closure stocks prepared", flush=True)
    stocks = {}
    for cf in {key[0] for key in buckets}:
        running = Counter()
        for year in range(1989, end_year + 1):
            running.update(closed_deltas.get((cf, year - 1), {}))
            if (cf, year) in buckets:
                stocks[(cf, year)] = running.copy()

    # Check that the county/year join did not duplicate any PA category amounts.
    for key, b in buckets.items():
        if b.counts["pa_eligible_records"]:
            categories = list(PA_CATEGORIES.values()) + ["other_category"]
            if sum(b.counts[f"pa_eligible_{label}_records"] for label in categories) != b.counts["pa_eligible_records"]:
                raise ValueError(f"PA category reconciliation failed: {key}")
            total = b.sums.get("pa_eligible_federal_share_obligated_nominal_usd", Sum()).total
            categorized = sum((b.sums.get(f"pa_eligible_{label}_federal_share_nominal_usd", Sum()).total for label in categories), Decimal(0))
            if total != categorized:
                raise ValueError(f"PA financial reconciliation failed: {key}")
    rows_by_year = Counter(year for cf, year in buckets)
    if not buckets:
        raise ValueError("No flood county/year observations in selected years")
    controls["nri"]["matched_county_years"] = sum(cf in nri for cf, year in buckets)
    controls["nri"]["unmatched_county_years"] = sum(cf not in nri for cf, year in buckets)
    meta = {
        "inputs": [str(p) for p in inputs], "outputs": [str(p) for p in outputs],
        "parameters": {"start_year": start_year, "end_year": end_year, "cpi_base_year": base_year,
                       "maximum_source_year": max_source_year, "time_resolution": "year"},
        "grain": "one row per observed flood county and fixed-EST event-start calendar year",
        "rows": len(buckets), "columns": len(schema.columns),
        "rows_by_year": dict(sorted(rows_by_year.items())), "source_records": dict(source_counts),
        "coverage": {key: dict(value) for key, value in controls.items()},
        "omitted_or_unallocated_nominal_controls": {key: {"known_records": value.known, "missing_records": value.missing, "amount": value.display()} for key, value in omissions.items()},
        "annual_cpi": {str(year): {"value": str(cpi[year]), "sum_of_12_observations": str(cpi_totals[year]),
                                  "interpolated_observations": cpi_fill[year]} for year in sorted(cpi)},
        "hma_duplicate_business_identifiers": {key: count for key, count in identifiers.items() if count > 1},
        "methods_and_limits": [
            "Local inputs only; no additional dataset, API or disaster-declaration interval is used.",
            "A county/year match provides context, not a causal/event-to-project attribution. NOAA episode IDs are not FEMA disaster numbers.",
            "The flood base is damage-screened. Absent county-years are not synthesized as zero-flood observations; noncounty records are counted in omissions.",
            "Source cleaning filled one missing damage component with zero. That imputation cannot be reversed here; casualty missingness is retained.",
            "PA geography is applicant location. Statewide/missing locations are unallocated even if a county code is populated.",
            "PA amounts are cumulative snapshot values of declaration-year cohorts, not annual disbursements. Exact Flood incidents and flood-capable incidents are separate; neither assigns all spending to NOAA flood events.",
            "HMA fiscal-year cohorts, first/latest approval, initial obligation and closure-year records have different meanings and are not additive.",
            "All statuses remain in HMA cohort counts. Awarded/closed and flood-related filters govern money and milestone summaries.",
            "Flood-related HMA means an explicitly mapped physical flood activity or FMA/SRL/RFC membership. Other potentially relevant measures are not assumed to be flood-specific.",
            "HMA id is the record key. Repeated projectIdentifier values are retained and flagged, not arbitrarily collapsed.",
            "HMA county names are resolved only when a state/name alias uniquely maps to a county code. Conflicts and statewide projects remain unallocated.",
            "Multicounty HMA coverage can appear in multiple counties, but project money is included only for single-county scope. Coverage and activity counts are not nationally additive.",
            "Current cohort/first-approval activity counts and prior-closure activity stocks are separate, so newly approved strategies are not confused with earlier mitigation context.",
            "Prior-year closure stocks include pre-1999 projects. Closure in the same calendar year is not assumed to precede that year's floods. Closure is a fiscal milestone, not verified protection at every flood point.",
            "Impossible years and reversed YEAR ordering are flagged and excluded from milestone/prior-closure features; ties within a year are not ordered.",
            "Positive BCR median is reported descriptively; it is not realized effectiveness and no model is trained.",
            "Flood CPI uses annual means after the existing adjacent-observation missing-value rule. Project monetary snapshots retain nominal values because precise spending price years cannot be reconstructed.",
            "NRI is the full authoritative static snapshot, including blanks, with separate IFLD/CFLD columns; it is not a historical social panel and is never summed over years.",
            "No matched project records does not establish no government action. Observation flags and blank money values preserve this distinction.",
            "Current PA/HMA/NRI snapshots are not decision-time observations; their fields must not be treated as leakage-free historical predictors.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for final in outputs:
            with tempfile.NamedTemporaryFile(dir=output_dir, prefix=".annual-", suffix=final.suffix, delete=False) as temp:
                staged.append(Path(temp.name))
        with staged[0].open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(schema.columns))
            writer.writeheader()
            for (cf, year), b in sorted(buckets.items()):
                record = dict.fromkeys(schema.columns, "")
                for name in schema.counts:
                    record[name] = b.counts[name]
                for name in schema.money:
                    total = b.sums.get(name, Sum())
                    record[name] = total.display()
                    record[name + "_known_records"] = total.known
                    record[name + "_missing_records"] = total.missing
                for name, values in b.sets.items():
                    if name in record:
                        record[name] = len(values)
                county = nri.get(cf, {})
                record.update(county_fips=cf, analysis_year=year, county_name=county.get("COUNTY", "") or ";".join(sorted(b.sets["area_names"])),
                              nri_match_status="matched" if county else "unmatched_county_fips", cpi_base_year=base_year,
                              cpi_source_interpolated_observations=cpi_fill[year], cpi_base_interpolated_observations=cpi_fill[base_year],
                              pa_county_year_record_observed=int(b.counts["pa_records"] > 0),
                              hma_fiscal_cohort_record_observed=int(b.counts["hma_fy_records"] > 0))
                for source in NRI_COMMON:
                    record["nri_" + source.lower()] = county.get(source, "")
                for prefix in ("IFLD", "CFLD"):
                    for suffix in NRI_SUFFIXES:
                        record["nri_" + prefix.lower() + "_" + suffix.lower()] = county.get(prefix + "_" + suffix, "")
                duration = b.values["duration"]
                record["flood_mean_duration_hours"] = str(sum(duration) / len(duration)) if duration else ""
                record["flood_max_duration_hours"] = str(max(duration)) if duration else ""
                record["flood_cause_missing_records"] = b.counts["cause:"]
                for output, prefix in (("flood_cause_counts", "cause:"), ("pa_status_counts", "pa_status:"),
                                       ("pa_process_step_counts", "pa_step:"), ("hma_fy_status_counts", "hma_status:")):
                    record[output] = json.dumps({name[len(prefix):]: count for name, count in sorted(b.counts.items()) if name.startswith(prefix) and name[len(prefix):]}, ensure_ascii=False, separators=(",", ":"))
                record["hma_fy_flood_awarded_positive_bcr_median"] = str(median(b.values["bcr"])) if b.values["bcr"] else ""
                record.update(stocks.get((cf, year), {}))
                writer.writerow(record)
        with staged[1].open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["field", "source", "definition", "units", "role"])
            writer.writeheader()
            writer.writerows(schema.columns.values())
        staged[2].write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for temp, final in zip(staged, outputs):
            temp.replace(final)
    finally:
        for temp in staged:
            if temp.exists():
                temp.unlink()
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flood", type=Path, default=ROOT / "data/processed_data/flood_final.csv")
    parser.add_argument("--pa", type=Path, default=ROOT / "PublicAssistanceFundedProjectsDetails.csv")
    parser.add_argument("--hma", type=Path, default=ROOT / "HazardMitigationAssistanceProjects.csv")
    parser.add_argument("--nri", type=Path, default=ROOT / "NRI_Table_Counties.csv")
    parser.add_argument("--cpi", type=Path, default=ROOT / "CPIAUCSL.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/analyzed_data/flood_policy_annual")
    parser.add_argument("--start-year", type=int, default=1999)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--base-year", type=int, default=2025)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = merge_annual(args.flood, args.pa, args.hma, args.nri, args.cpi, args.output_dir,
                            start_year=args.start_year, end_year=args.end_year, base_year=args.base_year,
                            overwrite=args.overwrite)
    print(f"Saved {metadata['rows']:,} rows, {metadata['columns']} fields, {args.start_year}-{args.end_year}")
    print(metadata["outputs"][0])


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as error:
        print(f"Annual merge failed: {error}", file=sys.stderr)
        raise SystemExit(1)
