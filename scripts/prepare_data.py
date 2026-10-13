"""Download and verify the unmodified official ETTh1 CSV (standard library only).

The complete source file is retained. Training code must use the documented
14,400-point TFB/ETT benchmark interval rather than repartition all 17,420 rows.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "1d16c8f4f943005d613b5bc962e9eeb06058cf07"
SOURCE_PATH = "ETT-small/ETTh1.csv"
SOURCE_URL = (
    f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/{SOURCE_COMMIT}/{SOURCE_PATH}"
)
FALLBACK_URL = (
    f"https://github.com/zhouhaoyi/ETDataset/raw/{SOURCE_COMMIT}/{SOURCE_PATH}"
)
EXPECTED_SHA256 = "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"
EXPECTED_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
EXPECTED_ROWS = 17420


def validate_bytes(payload: bytes) -> dict:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError(f"ETTh1 SHA256 mismatch: expected {EXPECTED_SHA256}; received {digest}")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if reader.fieldnames != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
    rows = list(reader)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} observations, received {len(rows)}")
    times = []
    for i, row in enumerate(rows):
        if set(row) != set(EXPECTED_COLUMNS):
            raise ValueError(f"Malformed CSV row {i}")
        timestamp = datetime.fromisoformat(row["date"])
        if times and timestamp - times[-1] != timedelta(hours=1):
            raise ValueError(f"Non-hourly or duplicate timestamp at observation {i}")
        times.append(timestamp)
        for name in EXPECTED_COLUMNS[1:]:
            if row[name] is None or not row[name].strip():
                raise ValueError(f"Missing value at observation {i}, variable {name}")
            if not math.isfinite(float(row[name])):
                raise ValueError(f"Non-finite value at observation {i}, variable {name}")
    splits = {}
    for name, start, end in [("train", 0, 8640), ("validation", 8640, 11520), ("test", 11520, 14400)]:
        splits[name] = {
            "start_inclusive": start,
            "end_exclusive": end,
            "length": end - start,
            "first_timestamp": rows[start]["date"],
            "last_timestamp": rows[end - 1]["date"],
        }
    return {
        "dataset": "ETTh1",
        "source_repository": "https://github.com/zhouhaoyi/ETDataset",
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "sha256": digest,
        "bytes": len(payload),
        "rows": len(rows),
        "columns": EXPECTED_COLUMNS,
        "variables": EXPECTED_COLUMNS[1:],
        "frequency": "1h",
        "first_timestamp": rows[0]["date"],
        "last_timestamp": rows[-1]["date"],
        "checks": {
            "pinned_sha256_matches": True,
            "expected_row_count": True,
            "expected_column_order": True,
            "strict_hourly_continuity": True,
            "unique_timestamps": True,
            "finite_numeric_values": True,
            "missing_values": 0,
        },
        "benchmark": {
            "protocol": "TFB ETTh1 Table 5; classical ETT 12/4/4 x 30-day periods",
            "used_rows": 14400,
            "retained_but_unused_rows": len(rows) - 14400,
            "split_ratio": "6:2:2",
            "tfb_tv_ratio": 0.8,
            "tfb_train_ratio_in_tv": 0.75,
            "splits": splits,
            "note": "Split intervals describe prediction targets. Validation/test inputs may use preceding observed history. No data truncation is written to the source CSV.",
            "evidence": [
                "https://www.vldb.org/pvldb/vol17/p2363-hu.pdf (Table 5, page 6)",
                "vendor/TFB/config/rolling_forecast_config.json",
                "https://github.com/thuml/iTransformer/blob/main/data_provider/data_loader.py (Dataset_ETT_hour)",
            ],
            "tfb_preprocessed_archive_byte_comparison": False,
        },
    }


def download() -> tuple[bytes, str]:
    errors = []
    for url in (SOURCE_URL, FALLBACK_URL):
        for attempt in range(2):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ETTh1-reproducibility/1.0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                validate_bytes(payload)
                return payload, url
            except (OSError, urllib.error.URLError, ValueError) as exc:
                errors.append(f"{url}, attempt {attempt + 1}: {exc}")
    raise RuntimeError("Official ETTh1 download failed:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ETTh1.csv")
    parser.add_argument("--force-download", action="store_true", help="Download again and verify before replacing the local file")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    download_url = None
    if output.exists() and not args.force_download:
        payload = output.read_bytes()
    else:
        payload, download_url = download()
        temporary = output.with_suffix(".csv.part")
        temporary.write_bytes(payload)
        temporary.replace(output)
    manifest = validate_bytes(payload)
    manifest["file"] = output.name
    manifest["verified_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["downloaded_in_this_invocation"] = download_url is not None
    if download_url:
        manifest["download_url_used"] = download_url
    manifest_path = output.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {output}: {manifest['rows']} rows, 7 variables, hourly, no missing values")
    print(f"SHA256: {manifest['sha256']}")
    print(f"TFB benchmark: first 14400 rows; train/validation/test = 8640/2880/2880")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
