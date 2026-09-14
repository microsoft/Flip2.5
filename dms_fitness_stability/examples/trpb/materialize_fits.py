"""Expose staged Amulet TrpB results at canonical external-data paths."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    os.environ.get("DMS_FITNESS_STABILITY_DATA", "/data/data_dms_stability")
)


def materialize_fits(
    *,
    amlt_root: Path,
    data_root: Path,
    allow_missing: bool,
) -> None:
    manifest = json.loads((HERE / "fit_jobs.json").read_text(encoding="utf-8"))
    experiment = manifest["experiment"]
    sources = {
        spec["job"]: amlt_root
        / experiment
        / spec["job"]
        / spec["remote_result"]
        for spec in manifest["fits"]
    }
    missing = [f"{job}: {path}" for job, path in sources.items() if not path.is_file()]
    if missing and not allow_missing:
        raise FileNotFoundError(
            "Missing staged Amulet fits:\n" + "\n".join(f"  {item}" for item in missing)
        )

    materialized = 0
    for spec in manifest["fits"]:
        source = sources[spec["job"]]
        if not source.is_file():
            print(f"missing: {spec['job']}")
            continue
        destination = (
            data_root
            / "trpb"
            / "fits"
            / spec["library"]
            / spec["canonical_filename"]
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.samefile(source):
                print(f"linked: {destination}")
                continue
            raise FileExistsError(
                f"Refusing to replace non-matching canonical fit: {destination}"
            )
        try:
            destination.hardlink_to(source)
            action = "linked"
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copy2(source, destination)
            action = "copied"
        materialized += 1
        print(f"{action}: {destination} <- {source}")
    print(f"Materialized {materialized} new fit(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amlt-root", type=Path, default=Path("/data/amlt"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    materialize_fits(
        amlt_root=args.amlt_root,
        data_root=args.data_root,
        allow_missing=args.allow_missing,
    )


if __name__ == "__main__":
    main()
