"""Create a reproducible first-pass MIDAS station screen around London.

This is an acquisition aid, not a rural/urban classification. Candidate sites
must still be checked against station history, exposure and local land cover.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


LONDON_LAT = 51.5074
LONDON_LON = -0.1278


def table_after_data(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", errors="replace") as file:
        rows = list(csv.reader(file))
    marker = next(index for index, row in enumerate(rows) if row == ["data"])
    header = rows[marker + 1]
    return [dict(zip(header, row)) for row in rows[marker + 2:] if row]


def distance_and_bearing(lat: float, lon: float) -> tuple[float, float]:
    """Return great-circle distance (km) and initial bearing from London."""
    lat1, lat2 = map(math.radians, [LONDON_LAT, lat])
    delta_lat = lat2 - lat1
    delta_lon = math.radians(lon - LONDON_LON)
    a = (math.sin(delta_lat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2)
    distance = 6371.0 * 2 * math.asin(math.sqrt(a))
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return distance, (math.degrees(math.atan2(y, x)) + 360) % 360


def sector(bearing: float) -> str:
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return labels[int((bearing + 22.5) // 45) % 8]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference-year", type=int, default=2021)
    parser.add_argument("--max-distance-km", type=float, default=100)
    parser.add_argument("--max-elevation-m", type=float, default=150)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    output = []
    for row in table_after_data(args.metadata):
        try:
            latitude = float(row["station_latitude"])
            longitude = float(row["station_longitude"])
            elevation = float(row["station_elevation"])
            first_year = int(row["first_year"])
            last_year = int(row["last_year"])
        except (KeyError, TypeError, ValueError):
            continue
        distance, bearing = distance_and_bearing(latitude, longitude)
        if (distance > args.max_distance_km or elevation > args.max_elevation_m
                or last_year < args.reference_year):
            continue
        output.append({
            "station_id": row["src_id"],
            "station_name": row["station_name"],
            "distance_from_central_london_km": round(distance, 2),
            "direction_from_central_london": sector(bearing),
            "latitude": latitude,
            "longitude": longitude,
            "elevation_m": elevation,
            "first_year": first_year,
            "last_year": last_year,
            "record_span_years": last_year - first_year + 1,
            "needs_hourly_file_check": True,
            "needs_station_history_check": True,
            "needs_landcover_exposure_check": True,
            "classification_status": "candidate only",
        })
    output.sort(key=lambda row: row["distance_from_central_london_km"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} station candidates to {args.output}")


if __name__ == "__main__":
    main()
