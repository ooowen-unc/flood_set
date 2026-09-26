"""Check final filtering and EST conversion using small CSV fixtures."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from clean_final import clean_file
from clean_flood import run
from test_clean_flood import event as raw_event, death, location


def event(event_id="1", zone="CST-6", begin="2025-07-01T12:00:00", end="2025-07-01T14:00:00"):
    return {"event_id": event_id, "event_type": "Flood", "year": "2025", "begin_time_local": begin,
            "end_time_local": end, "timezone": zone, "duration_hours": "2.0",
            "county_fips": "01001", "core_fields_valid": "1",
            "damage_property_usd": "5000", "damage_crops_usd": "0",
            "quality_flags": "county_code_unverified|timezone_not_normalized_to_utc"}


class FinalCleaningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source.csv"
        self.output = self.root / "flood_final.csv"

    def write(self, rows):
        with self.source.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def read(self):
        with self.output.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_summer_conversion_uses_reported_standard_zone(self):
        self.write([event()])
        clean_file(self.source, self.output)
        row = self.read()[0]
        self.assertEqual(row["begin_time_est"], "2025-07-01T13:00:00-05:00")
        self.assertEqual(row["end_time_est"], "2025-07-01T15:00:00-05:00")
        self.assertEqual(row["timezone"], "EST-5")
        self.assertEqual(row["source_timezone"], "CST-6")
        self.assertNotIn("quality_flags", row)
        self.assertEqual(row["county_fips"], "01001")
        self.assertEqual(row["duration_hours"], "2.0")
        self.assertNotIn("begin_time_local", row)

    def test_all_four_flood_types_preserved_in_final(self):
        source_dir = self.root / "raw"
        source_dir.mkdir()
        event_types = ("Flash Flood", "Flood", "Coastal Flood", "Lakeshore Flood", "Hail")
        details = [raw_event(str(index), kind, 2025)
                   for index, kind in enumerate(event_types, 1)]
        for kind, rows, template in (("details", details, details[0]),
                                     ("fatalities", [], death()), ("locations", [], location())):
            path = source_dir / f"StormEvents_{kind}-ftp_v1.0_d2025_c20260918.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(template))
                writer.writeheader()
                writer.writerows(rows)
        processed = self.root / "processed"
        with redirect_stderr(io.StringIO()):
            summary = run(source_dir, processed, 2025, 2025)
        clean_file(processed / "flood_2025_2025.csv", self.output)
        result = self.read()
        self.assertEqual([row["event_id"] for row in result], ["1", "2", "3", "4"])
        self.assertEqual([row["event_type"] for row in result], list(event_types[:4]))
        expected = [name.replace("_local", "_est") for name in summary["output_columns"]
                    if name != "quality_flags"]
        expected.insert(expected.index("timezone") + 1, "source_timezone")
        self.assertEqual(list(result[0]), expected)
        self.assertTrue(all(row["timezone"] == "EST-5" for row in result))
        self.assertTrue(all(row["damage_crops_usd"] == "0" for row in result))

    def test_missing_event_type_aborts_without_overwriting_output(self):
        self.write([event()])
        clean_file(self.source, self.output)
        original = self.output.read_bytes()
        row = event()
        del row["event_type"]
        self.write([row])
        with self.assertRaisesRegex(ValueError, "event_type"):
            clean_file(self.source, self.output, overwrite=True)
        self.assertEqual(self.output.read_bytes(), original)

    def test_all_observed_timezone_labels_have_known_conversion(self):
        expected_hours = {
            "EST": 12, "EST-5": 12, "CST": 13, "CST-6": 13,
            "MST": 14, "MST-7": 14, "PST": 15, "PST-8": 15,
            "AKST-9": 16, "HST": 17, "HST-10": 17,
            "AST": 11, "AST-4": 11, "SST": 18, "SST-11": 18, "GST10": 21,
            "EDT-4": 11, "CDT-5": 12, "MDT-6": 13, "PDT-7": 14,
            "AKDT-8": 15, "HDT-9": 16,
        }
        self.write([event(str(index), zone) for index, zone in enumerate(expected_hours, 1)])
        summary = clean_file(self.source, self.output)
        for row in self.read():
            zone = row["source_timezone"]
            date = "2025-06-30" if zone == "GST10" else "2025-07-01"
            self.assertEqual(row["begin_time_est"], f"{date}T{expected_hours[zone]:02d}:00:00-05:00")
        self.assertEqual(summary["rows"], len(expected_hours))

    def test_guam_can_move_to_previous_year_without_changing_source_year(self):
        self.write([event(zone="GST10", begin="2025-01-01T02:00:00", end="2025-01-01T04:00:00")])
        summary = clean_file(self.source, self.output)
        row = self.read()[0]
        self.assertEqual(row["begin_time_est"], "2024-12-31T11:00:00-05:00")
        self.assertEqual(row["year"], "2025")
        self.assertEqual(summary["events_with_year_change"], 1)

    def test_pacific_time_can_move_to_next_year(self):
        self.write([event(zone="PST", begin="2025-12-31T22:00:00", end="2026-01-01T00:00:00")])
        clean_file(self.source, self.output)
        self.assertEqual(self.read()[0]["begin_time_est"], "2026-01-01T01:00:00-05:00")

    def test_bad_zone_or_timestamp_aborts_without_partial_output(self):
        for zone, begin in (("XYZ", "2026-07-01T12:00:00"),
                            ("CST-5", "2026-07-01T12:00:00"),
                            ("EST", "2026-07-01"), ("EST", ""),
                            ("EST", "2026-07-01T12:00:00-05:00")):
            with self.subTest(zone=zone, begin=begin):
                self.write([event(), event("2", zone=zone, begin=begin)])
                with self.assertRaises(ValueError):
                    clean_file(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_duplicate_id_or_duration_conflict_aborts(self):
        bad = event("2")
        bad["duration_hours"] = "3.0"
        for rows in ([event(), event()], [event(), bad]):
            self.write(rows)
            with self.assertRaises(ValueError):
                clean_file(self.source, self.output)
            self.assertFalse(self.output.exists())

    def test_source_and_existing_output_are_protected(self):
        self.write([event()])
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            clean_file(self.source, self.source, overwrite=True)
        clean_file(self.source, self.output)
        output = self.output.read_bytes()
        with self.assertRaises(ValueError):
            clean_file(self.source, self.output)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.output.read_bytes(), output)

    def test_year_core_and_damage_rules_keep_explicit_zero(self):
        rows = []
        for event_id, year, valid, property_damage, crop_damage in (
            ("1", "1996", "1", "10", "10"),
            ("2", "1997", "1", "10", "10"),
            ("3", "1998", "1", "10", "10"),
            ("4", "2026", "1", "10", "10"),
            ("5", "1999", "1", "", "  "),
            ("6", "2000", "1", "", "125"),
            ("7", "2001", "1", "50", "  "),
            ("8", "2025", "1", "0", "0"),
            ("9", "2005", "0", "1", "1"),
            ("10", "2006", "1", "25", "60"),
            ("11", "1999", "1", "", "0"),
            ("12", "2025", "1", "0", ""),
        ):
            row = event(event_id)
            row.update(year=year, core_fields_valid=valid,
                       damage_property_usd=property_damage, damage_crops_usd=crop_damage)
            if valid == "0":
                row["begin_time_local"] = ""
            rows.append(row)
        self.write(rows)
        summary = clean_file(self.source, self.output)
        result = self.read()
        self.assertEqual([row["event_id"] for row in result], ["3", "6", "7", "8", "10", "11", "12"])
        self.assertEqual([(row["damage_property_usd"], row["damage_crops_usd"]) for row in result],
                         [("10", "10"), ("0", "125"), ("50", "0"), ("0", "0"), ("25", "60"), ("0", "0"), ("0", "0")])
        self.assertEqual(summary["input_rows"], 12)
        self.assertEqual(summary["removed_year_outside_range"], 3)
        self.assertEqual(summary["removed_core_invalid"], 1)
        self.assertEqual(summary["removed_both_damage_missing"], 1)
        self.assertEqual(summary["filled_property_damage"], 2)
        self.assertEqual(summary["filled_crop_damage"], 2)

    def test_quality_flags_are_optional_and_invalid_core_flag_aborts(self):
        row = event()
        del row["quality_flags"]
        self.write([row])
        clean_file(self.source, self.output)
        self.assertNotIn("quality_flags", self.read()[0])
        original = self.output.read_bytes()
        row["core_fields_valid"] = "unknown"
        self.write([row])
        with self.assertRaisesRegex(ValueError, "invalid core_fields_valid"):
            clean_file(self.source, self.output, overwrite=True)
        self.assertEqual(self.output.read_bytes(), original)

    def test_overwrite_replaces_only_after_success(self):
        self.write([event()])
        clean_file(self.source, self.output)
        original = self.output.read_bytes()
        self.write([event(), event("2", zone="XYZ")])
        with self.assertRaises(ValueError):
            clean_file(self.source, self.output, overwrite=True)
        self.assertEqual(self.output.read_bytes(), original)
        self.write([event("2")])
        clean_file(self.source, self.output, overwrite=True)
        self.assertEqual([row["event_id"] for row in self.read()], ["2"])


if __name__ == "__main__":
    unittest.main()
