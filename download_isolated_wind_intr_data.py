import os
import requests
import tarfile
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# URMP Trumpet Download & Extraction Script
# ==============================================================================
# This script manages the acquisition of trumpet tracks from the URMP dataset.
# Since the dataset is large (~12GB), it supports manual path specification
# for users who downloaded the archive to an external drive.
# ==============================================================================

DATASET_URL = "https://datadryad.org/downloads/file_stream/99348"
DEFAULT_DATA_DIR = "data/raw/urmp"
ARCHIVE_NAME = "Dataset.tar.gz"

def download_file(url, target_path):
    print(f"--- Downloading URMP Dataset from Dryad ---")
    print(f"Warning: This is a ~12GB download.")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    with open(target_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=ARCHIVE_NAME) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def extract_and_organize(archive_path, output_dir, source_dir=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if source_dir:
        print(f"--- Using existing source directory: {source_dir} ---")
        search_dir = Path(source_dir)
    else:
        archive_path = Path(archive_path)
        print(f"--- Extracting {archive_path.name} to {output_dir} ---")
        # Extract
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=output_dir)
        search_dir = output_dir

    print("--- Isolating Instrument Tracks ---")
    
    # Instruments to isolate and their URMP codes
    # User requested: Oboe, Saxophone, Bassoon, Trombone, French Horn, and Tuba
    # Mapping based on URMP structure: ob, sax, bn, tbn, hn, tba
    instr_map = {
        "trumpet": "tpt",
        "oboe": "ob",
        "sax": "sax",
        "bassoon": "bn",
        "trombone": "tbn",
        "horn": "hn",
        "tuba": "tba"
    }

    for name, code in instr_map.items():
        instr_dir = output_dir / f"{name}_only"
        instr_dir.mkdir(exist_ok=True)
        
        # Search for files with the instrument code
        files = list(search_dir.glob(f"**/AuSep_*_{code}_*.wav"))
        
        # Fallback to full name if code yields nothing (unlikely in URMP)
        if not files:
            files = list(search_dir.glob(f"**/AuSep_*_{name}_*.wav"))
            
        if files:
            print(f"Found {len(files)} tracks for {name}. Copying...")
            for f in files:
                shutil.copy2(f, instr_dir / f.name)
        else:
            print(f"No tracks found for {name}.")
        
    print(f"\n[SUCCESS] Task complete.")
    print(f"Isolated files are in {output_dir.absolute()}")

def main():
    parser = argparse.ArgumentParser(description="Download and extract URMP trumpet data.")
    parser.add_argument("--local_archive", type=str, help="Path to a manually downloaded Dataset.tar.gz")
    parser.add_argument("--source_dir", type=str, help="Path to an already expanded URMP dataset directory")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory to store extracted/isolated files")
    parser.add_argument("--download", action="store_true", help="Force download even if archive not found")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir

    if args.source_dir:
        extract_and_organize(None, output_dir, source_dir=args.source_dir)
        return

    archive_path = args.local_archive if args.local_archive else os.path.join(output_dir, ARCHIVE_NAME)

    # 1. Handle Download
    if not os.path.exists(archive_path):
        if args.download or not args.local_archive:
            download_file(DATASET_URL, archive_path)
        else:
            print(f"Error: Archive not found at {archive_path}")
            print("Please provide --local_archive, --source_dir, or use --download.")
            return
    else:
        print(f"Found archive at {archive_path}")

    # 2. Extract and Organize
    extract_and_organize(archive_path, output_dir)

if __name__ == "__main__":
    main()
