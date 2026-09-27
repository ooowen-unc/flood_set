"""Shared schema and paths; importing this file never trains a model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/analyzed_data/flood_policy_annual/flood_policy_county_year_1999_2025.csv"
SPLITS = ROOT / "data/modeling/splits"
ARTIFACTS = ROOT / "data/modeling/models"
TARGET = "pa_flood_incident_federal_share_obligated_nominal_usd"
MEASURES = ("acquisition", "drainage", "elevation", "flood_control")
MEASURE_NAMES = dict(zip(MEASURES, ("Property acquisition and floodplain retreat", "Drainage and stormwater management", "Building elevation", "Flood control infrastructure")))
LABELS = tuple(f"hma_flood_first_approval_year_{m}_records" for m in MEASURES)
PRIOR = tuple(f"hma_{m}_closed_before_year_records" for m in MEASURES)
BACKGROUND = ("nri_population", "nri_buildvalue", "nri_agrivalue", "nri_area",
              "nri_sovi_score", "nri_resl_score", "nri_ifld_risks", "nri_ifld_afreq")
TYPES = ("flood_flash_flood_records", "flood_inland_flood_records")
KEYS = ("county_fips", "analysis_year")
REQUIRED = (*KEYS, "county_name", "nri_match_status", TARGET, *LABELS, *PRIOR,
            *BACKGROUND, *TYPES, "flood_material_loss_adjusted_usd",
            "flood_deaths_direct", "flood_deaths_indirect",
            "flood_injuries_direct", "flood_injuries_indirect",
            "hma_physical_flood_closed_before_year_records", "nri_nri_ver")
