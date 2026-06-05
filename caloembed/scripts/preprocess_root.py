"""CLI to convert ROOT files to HDF5.

Single file:
    python -m caloembed.scripts.preprocess_root \\
        --input  /path/to/histo_921.root \\
        --output /path/to/hdf5/histo_921.h5

Full directory (preserves filenames, .root → .h5):
    python -m caloembed.scripts.preprocess_root \\
        --input-dir  /vols/cms/mm1221/cms/Data/100k/root/ \\
        --output-dir /vols/cms/mm1221/cms/Data/100k/hdf5/ \\
        --jobs 8

Already-converted files are skipped unless --overwrite is given.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def _process_one(args_tuple):
    """Worker function for multiprocessing (must be top-level, picklable)."""
    root_path, hdf5_path, compression = args_tuple
    from caloembed.data.preprocess import convert_file
    try:
        result = convert_file(root_path, hdf5_path, compression=compression)
        return {"status": "ok", **result}
    except Exception as e:
        return {"status": "error", "root_file": str(root_path), "error": str(e)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert HGCal ROOT files (ticlDumper/simtrackstersCP) to HDF5"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input",     help="Single ROOT file to convert")
    mode.add_argument("--input-dir", help="Directory of ROOT files to convert")

    parser.add_argument("--output",     help="Output HDF5 path (single-file mode)")
    parser.add_argument("--output-dir", help="Output directory (batch mode)")
    parser.add_argument("--jobs", "-j", type=int, default=1,
                        help="Number of parallel workers (batch mode only)")
    parser.add_argument("--compression", choices=["gzip", "lzf"], default="gzip",
                        help="HDF5 compression filter (default: gzip)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-convert files that already have an HDF5 output")

    args = parser.parse_args(argv)

    if args.input:
        root_path = Path(args.input)
        if args.output:
            hdf5_path = Path(args.output)
        else:
            hdf5_path = root_path.with_suffix(".h5")

        if hdf5_path.exists() and not args.overwrite:
            print(f"Skipping (already exists): {hdf5_path}  — use --overwrite to force")
            sys.exit(0)

        print(f"Converting {root_path} → {hdf5_path}")
        from caloembed.data.preprocess import convert_file
        result = convert_file(root_path, hdf5_path, compression=args.compression)
        print(json.dumps(result, indent=2))
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    root_files = sorted(input_dir.glob("*.root"))
    if not root_files:
        print(f"No .root files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    tasks = []
    for rf in root_files:
        hf = output_dir / rf.with_suffix(".h5").name
        if hf.exists() and not args.overwrite:
            print(f"Skip (exists): {hf.name}")
            continue
        tasks.append((str(rf), str(hf), args.compression))

    if not tasks:
        print("All files already converted. Use --overwrite to re-run.")
        return

    print(f"Converting {len(tasks)} files with {args.jobs} worker(s) ...")

    results = []
    if args.jobs == 1:
        for task in tasks:
            r = _process_one(task)
            results.append(r)
            status = "✓" if r["status"] == "ok" else "✗"
            name = Path(task[1]).name
            if r["status"] == "ok":
                print(f"  {status} {name}  [{r['n_events']} events, {r['n_lc_total']} LCs, {r['elapsed_s']:.1f}s]")
            else:
                print(f"  {status} {name}  ERROR: {r['error']}", file=sys.stderr)
    else:
        import multiprocessing
        with multiprocessing.Pool(processes=args.jobs) as pool:
            for r in pool.imap_unordered(_process_one, tasks):
                results.append(r)
                status = "✓" if r["status"] == "ok" else "✗"
                name = Path(r.get("hdf5_file", r.get("root_file", "?"))).name
                if r["status"] == "ok":
                    print(f"  {status} {name}  [{r['n_events']} events, {r['n_lc_total']} LCs, {r['elapsed_s']:.1f}s]")
                else:
                    print(f"  {status} {name}  ERROR: {r['error']}", file=sys.stderr)

    n_ok  = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    print(f"\nDone: {n_ok} converted, {n_err} errors")
    if n_err:
        sys.exit(1)


if __name__ == "__main__":
    main()
