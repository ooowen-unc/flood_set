"""Behavior checks using small CSV fixtures; never write to project data.

Run: python -B -m unittest discover -s scripts -p test_clean_flood.py -v
"""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from clean_flood import run


def event(event_id="10", event_type="Flood", year=2026, **changes):
    row = {
        "EVENT_ID": event_id, "EPISODE_ID": "20", "EVENT_TYPE": event_type,
        "YEAR": str(year), "STATE": "OHIO", "STATE_FIPS": "39",
        "CZ_TYPE": "C", "CZ_FIPS": "9", "CZ_NAME": "ATHENS", "CZ_TIMEZONE": "EST-5",
        "BEGIN_DATE_TIME": f"01-JAN-{year % 100:02d} 01:00:00",
        "END_DATE_TIME": f"01-JAN-{year % 100:02d} 03:00:00",
        "BEGIN_YEARMONTH": f"{year}01", "BEGIN_DAY": "1", "BEGIN_TIME": "100",
        "END_YEARMONTH": f"{year}01", "END_DAY": "1", "END_TIME": "300",
        "BEGIN_LAT": "39.3", "BEGIN_LON": "-82.2", "END_LAT": "39.4", "END_LON": "-82.3",
        "DAMAGE_PROPERTY": "5.00K", "DAMAGE_CROPS": "", "INJURIES_DIRECT": "0",
        "INJURIES_INDIRECT": "0", "DEATHS_DIRECT": "0", "DEATHS_INDIRECT": "0",
        "FLOOD_CAUSE": "Heavy Rain", "SOURCE": "Law Enforcement",
        "EVENT_NARRATIVE": 'Road flooded, sign read "closed".\nNo traffic.',
    }
    row.update(changes)
    return row


def death(fatality_id="30", **changes):
    row = {"EVENT_ID": "10", "FATALITY_ID": fatality_id, "FATALITY_TYPE": "D",
           "FATALITY_DATE": "01/01/2026 00:00:00", "FATALITY_AGE": "65",
           "FATALITY_SEX": "F", "FATALITY_LOCATION": "In Water"}
    row.update(changes)
    return row


def location(index="1", **changes):
    row = {"EVENT_ID": "10", "EPISODE_ID": "20", "LOCATION_INDEX": index,
           "LATITUDE": "39.3", "LONGITUDE": "-82.2"}
    row.update(changes)
    return row


class FloodCleaningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.input = self.root / "Archive"
        self.input.mkdir()
        self.output = self.root / "processed_data"

    def write(self, kind, year, rows, template, creation="20260918"):
        path = self.input / f"StormEvents_{kind}-ftp_v1.0_d{year}_c{creation}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(template))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def sources(self, year=2026, events=None, deaths=None, locations=None):
        self.write("details", year, events if events is not None else [event(year=year)], event(year=year))
        self.write("fatalities", year, deaths or [], death())
        self.write("locations", year, locations or [], location())

    def clean(self, start=2026, end=2026, dry_run=False):
        with redirect_stderr(io.StringIO()):
            return run(self.input, self.output, start, end, dry_run=dry_run)

    def output_rows(self, year=2026):
        with (self.output / f"flood_{year}.csv").open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_four_flood_types_and_no_join_multiplication(self):
        self.sources(events=[event(DEATHS_DIRECT="2"), event("11", "Flash Flood", DEATHS_DIRECT="1"),
                             event("12", "Coastal Flood"), event("13", "Lakeshore Flood"),
                             event("14", "Hail"), event("15", "Storm Surge/Tide")],
                     deaths=[death(), death("31"), death("32", EVENT_ID="11")],
                     locations=[location(), location("2", LATITUDE="39.4", LONGITUDE="-82.3"),
                                location("3", LATITUDE="39.4", LONGITUDE="-82.3"),
                                location(EVENT_ID="12"), location(EVENT_ID="13")])
        summary = self.clean()
        rows = self.output_rows()
        self.assertEqual([row["event_id"] for row in rows], ["10", "11", "12", "13"])
        self.assertEqual([row["event_type"] for row in rows],
                         ["Flood", "Flash Flood", "Coastal Flood", "Lakeshore Flood"])
        self.assertEqual(rows[0]["event_id"], "10")
        self.assertEqual(rows[0]["damage_property_usd"], "5000.00")
        self.assertEqual(rows[0]["deaths_direct"], "2")
        self.assertEqual(rows[0]["fatality_records"], "2")
        self.assertEqual(json.loads(rows[0]["location_points"]), [[39.3, -82.2], [39.4, -82.3]])
        self.assertEqual(rows[1]["fatality_records"], "1")
        self.assertEqual(json.loads(rows[2]["location_points"]), [[39.3, -82.2]])
        self.assertEqual(json.loads(rows[3]["location_points"]), [[39.3, -82.2]])
        self.assertEqual(summary["counts"]["flood_rows_selected"], 4)
        self.assertNotIn("EVENT_NARRATIVE", rows[0])

    def test_unknown_amounts_remain_distinct_from_explicit_zero(self):
        self.sources(events=[event(DAMAGE_PROPERTY="", DAMAGE_CROPS="0.00K"),
                             event("11", DAMAGE_PROPERTY="unknown", DAMAGE_CROPS="1.25M")])
        self.clean()
        rows = self.output_rows()
        self.assertEqual(rows[0]["damage_property_usd"], "")
        self.assertEqual(rows[0]["damage_crops_usd"], "0.00")
        self.assertEqual(rows[1]["damage_property_usd"], "")
        self.assertEqual(rows[1]["damage_crops_usd"], "1250000.00")
        self.assertIn("damage_property_invalid", rows[1]["quality_flags"])

    def test_conflicting_ids_excluded_but_disputed_time_retained_with_flag(self):
        self.sources(events=[event(), event(DAMAGE_PROPERTY="6K"),
                             event("11", END_DATE_TIME="01-JAN-26 05:00:00"), event("12")])
        summary = self.clean()
        rows = self.output_rows()
        self.assertEqual([row["event_id"] for row in rows], ["11", "12"])
        self.assertEqual([row["core_fields_valid"] for row in rows], ["0", "1"])
        self.assertEqual(summary["review_reasons"]["conflicting_duplicate_identifier"], 2)
        self.assertEqual(summary["review_reasons"]["invalid_core_fields"], 1)

    def test_child_conflicts_do_not_silently_override_death_totals(self):
        self.sources(events=[event(DEATHS_DIRECT="1")],
                     deaths=[death(), death(FATALITY_AGE="66")],
                     locations=[location(EPISODE_ID="99")])
        summary = self.clean()
        row = self.output_rows()[0]
        self.assertEqual(row["deaths_direct"], "")
        self.assertIn("deaths_direct_detail_mismatch", row["quality_flags"])
        self.assertEqual(row["fatality_records"], "0")
        self.assertEqual(summary["review_reasons"]["episode_id_conflict"], 1)

    def test_dry_run_has_no_output_and_existing_output_is_protected(self):
        self.sources()
        self.clean(dry_run=True)
        self.assertFalse(self.output.exists())
        self.clean()
        original = (self.output / "flood_2026.csv").read_bytes()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.clean()
        self.assertEqual((self.output / "flood_2026.csv").read_bytes(), original)

    def test_empty_year_header_and_latest_snapshot_and_zone_key(self):
        self.sources(year=2025, events=[event("11", "Hail", 2025)])
        self.sources(events=[event(CZ_TYPE="Z", CZ_FIPS="9")])
        self.write("details", 2026, [event("99")], event(), creation="20260101")
        self.clean(start=2025)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         ["flood_2025.csv", "flood_2025_2026.csv", "flood_2026.csv"])
        self.assertEqual(self.output_rows(2025), [])
        row = self.output_rows()[0]
        self.assertEqual(row["event_id"], "10")
        self.assertEqual(row["county_fips"], "")
        self.assertEqual(row["area_code"], "009")
        self.assertIn("requires_crosswalk", row["quality_flags"])
        self.assertEqual(row["year_is_partial"], "1")

    def test_combined_table_keeps_one_header_and_all_years(self):
        self.sources(year=2025, events=[event("11", year=2025)])
        self.sources()
        summary = self.clean(start=2025)
        with (self.output / "flood_2025_2026.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["event_id"] for row in rows], ["11", "10"])
        self.assertEqual([row["year"] for row in rows], ["2025", "2026"])
        self.assertEqual(reader.fieldnames, list(rows[0]))
        self.assertTrue(summary["storage_assessment"]["suitable_for_single_csv"])
        self.assertEqual(summary["storage_assessment"]["combined_rows"], 2)

    def test_nonstandard_quotes_in_unused_narrative_do_not_lose_flood_event(self):
        self.sources(events=[event(EVENT_NARRATIVE='S OF "7 MILE" BRIDGE')])
        path = next(self.input.glob("StormEvents_details*.csv"))
        text = path.read_text(encoding="utf-8").replace('""7 MILE""', '"7 MILE"')
        path.write_text(text, encoding="utf-8")
        self.clean()
        row = self.output_rows()[0]
        self.assertEqual(row["event_id"], "10")
        self.assertEqual(row["damage_property_usd"], "5000.00")
        self.assertIn("nonstandard_csv_quoting", row["quality_flags"])

    def test_ambiguous_multiline_corruption_aborts_without_output(self):
        self.sources()
        path = next(self.input.glob("StormEvents_details*.csv"))
        text = path.read_text(encoding="utf-8").replace('""closed""', '"closed"')
        path.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Column count mismatch"):
            self.clean()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
