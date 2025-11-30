#!/usr/bin/env python3
"""
Merge experiment results from multiple platforms into a single comparison CSV.
Usage: python3 merge_results.py <result_dir1> <result_dir2> ... -o output.csv
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any


def read_experiment_config(result_dir: Path) -> Dict[str, str]:
    """Read experiment configuration from result directory."""
    config_file = result_dir / "experiment_config.txt"
    config = {}
    
    if config_file.exists():
        with open(config_file) as f:
            for line in f:
                line = line.strip()
                if ":" in line and not line.startswith("="):
                    key, value = line.split(":", 1)
                    config[key.strip()] = value.strip()
    
    return config


def read_profile_summary(result_dir: Path) -> List[Dict[str, Any]]:
    """Read profile summary CSV from result directory."""
    csv_file = result_dir / "profile_outputs" / "profile_summary.csv"
    
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found")
        return []
    
    rows = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    return rows


def merge_results(result_dirs: List[Path], output_file: Path) -> None:
    """Merge results from multiple experiment directories."""
    all_rows = []
    
    print("Merging results from:")
    for result_dir in result_dirs:
        print(f"  - {result_dir}")
        
        # Read config to get platform info
        config = read_experiment_config(result_dir)
        platform = config.get("Platform", "unknown")
        hostname = config.get("Hostname", "unknown")
        timestamp = config.get("Timestamp", "unknown")
        
        # Read CSV data
        rows = read_profile_summary(result_dir)
        
        # Add platform metadata to each row
        for row in rows:
            row["platform"] = platform
            row["hostname"] = hostname
            row["timestamp"] = timestamp
            all_rows.append(row)
        
        print(f"    Found {len(rows)} entries from {platform}")
    
    if not all_rows:
        print("No data to merge!")
        return
    
    # Write merged CSV
    fieldnames = list(all_rows[0].keys())
    
    # Reorder columns to put platform info first
    priority_fields = ["platform", "hostname", "timestamp"]
    for field in reversed(priority_fields):
        if field in fieldnames:
            fieldnames.remove(field)
            fieldnames.insert(0, field)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    
    print(f"\nMerged CSV written to: {output_file}")
    print(f"Total entries: {len(all_rows)}")
    
    # Print summary by platform
    platforms = {}
    for row in all_rows:
        p = row.get("platform", "unknown")
        platforms[p] = platforms.get(p, 0) + 1
    
    print("\nEntries by platform:")
    for platform, count in platforms.items():
        print(f"  {platform}: {count}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge NSIGHT experiment results from multiple platforms"
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="Paths to experiment result directories"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("merged_results.csv"),
        help="Output CSV file (default: merged_results.csv)"
    )
    
    args = parser.parse_args()
    
    # Validate input directories
    valid_dirs = []
    for result_dir in args.result_dirs:
        if not result_dir.exists():
            print(f"Warning: Directory not found: {result_dir}")
            continue
        
        csv_file = result_dir / "profile_outputs" / "profile_summary.csv"
        if not csv_file.exists():
            print(f"Warning: No profile_summary.csv in {result_dir}")
            continue
        
        valid_dirs.append(result_dir)
    
    if not valid_dirs:
        print("Error: No valid result directories provided")
        return 1
    
    merge_results(valid_dirs, args.output)
    return 0


if __name__ == "__main__":
    exit(main())
