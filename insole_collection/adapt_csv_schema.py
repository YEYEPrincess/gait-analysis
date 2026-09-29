"""
Convert an insole CSV written with the `foot_side` / `chN` schema into the
`client` / `ch_N` schema that this repository's logger produces.

Some sessions (e.g. wuyu) were captured with a different collector that names
the foot column `foot_side` and the channels `ch0..ch252`, while
`common/logger.py` writes `client` and `ch_0..ch_252`. The values, units and
timestamps are identical, so only the header needs translating before
`offline_processing_for_3djoint.py` can read the file.

Column mapping:
    timestamp -> timestamp   (unchanged, Unix epoch seconds)
    foot_side -> client      (0 = left, 1 = right in both schemas)
    chN       -> ch_N        (same channel order)

Usage:
    python insole_collection/adapt_csv_schema.py wuyu
    python insole_collection/adapt_csv_schema.py wuyu --in path/to/wuyu.csv --out path/to/out.csv

Then run the normal pipeline:
    python insole_collection/offline_processing_for_3djoint.py wuyu
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Where offline_processing_for_3djoint.py looks for "<session>.csv".
CSV_ROOT = Path(r"d:\Desktop\UCL\Msc project\v1_Smart_insole_unet")

# Where this repo's collection scripts drop their captures.
COMMON_DIR = Path(__file__).resolve().parent / "common"

N_CHANNELS = 253

FOOT_COL_ALIASES = ("foot_side", "foot", "side")


def find_input(session: str) -> Path:
    for candidate in (COMMON_DIR / f"{session}.csv", CSV_ROOT / f"{session}.csv"):
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"No CSV found for session '{session}'. Looked in:\n"
        f"  {COMMON_DIR / f'{session}.csv'}\n"
        f"  {CSV_ROOT / f'{session}.csv'}\n"
        f"Pass an explicit path with --in."
    )


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)

    if "timestamp" not in cols:
        raise SystemExit(f"CSV has no 'timestamp' column. Columns start with: {cols[:6]}")

    if "client" in cols and any(c.startswith("ch_") for c in cols):
        print("Already in the client/ch_N schema - nothing to translate.")
        return df

    foot_col = next((c for c in FOOT_COL_ALIASES if c in cols), None)
    if foot_col is None:
        raise SystemExit(
            f"No foot column found (tried {FOOT_COL_ALIASES}). Columns start with: {cols[:6]}"
        )

    # Channels are `chN` with no underscore; keep them in numeric order rather
    # than the CSV's textual order so ch10 cannot land before ch2.
    channel_cols = [c for c in cols if re.fullmatch(r"ch\d+", c)]
    if not channel_cols:
        raise SystemExit(f"No 'chN' channel columns found. Columns start with: {cols[:6]}")

    channel_cols.sort(key=lambda c: int(c[2:]))
    if len(channel_cols) != N_CHANNELS:
        print(f"Warning: found {len(channel_cols)} channels, expected {N_CHANNELS}")

    feet = sorted(df[foot_col].dropna().unique().tolist())
    if not set(feet) <= {0, 0.0, 1, 1.0}:
        print(f"Warning: '{foot_col}' holds unexpected values {feet}; expected 0 (left) / 1 (right)")

    out = pd.DataFrame({
        "timestamp": df["timestamp"],
        "client": df[foot_col].astype(int),
    })
    renamed = df[channel_cols].rename(columns=lambda c: f"ch_{int(c[2:])}")
    out = pd.concat([out, renamed], axis=1)

    print(f"Mapped '{foot_col}' -> 'client' and {len(channel_cols)} 'chN' -> 'ch_N' columns")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", help="session name, e.g. wuyu")
    parser.add_argument("--in", dest="src", default=None, help="input CSV (default: auto-discover)")
    parser.add_argument("--out", dest="dst", default=None,
                        help=f"output CSV (default: {CSV_ROOT}\\<session>.csv)")
    args = parser.parse_args()

    src = Path(args.src) if args.src else find_input(args.session)
    dst = Path(args.dst) if args.dst else CSV_ROOT / f"{args.session}.csv"

    if dst.resolve() == src.resolve():
        raise SystemExit(f"Refusing to overwrite the input file in place: {src}")
    if dst.exists():
        print(f"Warning: overwriting existing {dst}")

    print(f"Reading  {src}")
    df = pd.read_csv(src)
    print(f"  {len(df)} rows x {len(df.columns)} columns")

    out = adapt(df)

    counts = out["client"].value_counts().sort_index()
    for client, n in counts.items():
        print(f"  client {client} ({'left' if client == 0 else 'right'}): {n} frames")
    span = out["timestamp"].iloc[-1] - out["timestamp"].iloc[0]
    print(f"  duration {span:.1f}s")

    dst.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False)
    print(f"Wrote    {dst}")
    print(f"\nNext:\n  python insole_collection/offline_processing_for_3djoint.py {args.session}")


if __name__ == "__main__":
    sys.exit(main())
