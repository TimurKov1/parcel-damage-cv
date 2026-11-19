"""Evaluate a fixed model and its thresholds on labeled multiview claims."""
import argparse
import json
from pathlib import Path

import joblib

from parcel_cv import features, load_manifest, route_claims, score_images, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        rows = load_manifest(args.manifest)
        bundle = joblib.load(args.model)
        inputs = [features(args.manifest.parent / row["path"]) for row in rows]
        views = score_images(rows, bundle["quality"].predict_proba(inputs)[:, 1],
                             bundle["damage"].predict_proba(inputs)[:, 1])
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({
        "all_photos": summarize(route_claims(views, bundle["low"], bundle["high"])),
        "first_photo": summarize(route_claims(views, bundle["low"], bundle["high"], first_only=True)),
    }, indent=2))


if __name__ == "__main__":
    main()
