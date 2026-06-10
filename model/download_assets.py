#!/usr/bin/env python3
"""Download Unitree G1 mesh assets from MuJoCo Menagerie on GitHub.

Downloads the STL files referenced by model/g1_raw.xml into model/assets/
so the full visual model can be rendered in the MuJoCo viewer.
"""
import os, re, urllib.request, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
RAW_XML = os.path.join(HERE, "g1_raw.xml")
ASSETS_DIR = os.path.join(HERE, "assets")

BASE_URL = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/assets"

# Parse STL filenames from g1_raw.xml
with open(RAW_XML) as f:
    xml = f.read()
stl_files = sorted(set(re.findall(r'file="([^"]+\.STL)"', xml)))

print(f"Found {len(stl_files)} mesh files referenced in g1_raw.xml")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Also download the scene.xml for reference
scene_url = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/scene.xml"

errors = []
for i, fname in enumerate(stl_files):
    dest = os.path.join(ASSETS_DIR, fname)
    if os.path.exists(dest):
        print(f"  [{i+1}/{len(stl_files)}] {fname} — already exists, skipping")
        continue
    url = f"{BASE_URL}/{fname}"
    print(f"  [{i+1}/{len(stl_files)}] {fname} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = os.path.getsize(dest) / 1024
        print(f"OK ({size_kb:.0f} KB)")
    except Exception as e:
        print(f"FAILED: {e}")
        errors.append(fname)

if errors:
    print(f"\n{len(errors)} files failed to download:")
    for f in errors:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"\nAll {len(stl_files)} mesh files downloaded to {ASSETS_DIR}")
