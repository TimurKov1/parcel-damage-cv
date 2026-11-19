"""Evaluate single-photo damage classification on real photographs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from parcel_cv import features, models


DATA_URL = ("https://raw.githubusercontent.com/chanllon/corrugated-cardboard-boxes-dataset/"
            "d014d6655c9998771078045e05d7eaf729f4ed38/")


def load_photos(index_path, image_directory, download=False):
    seen = set()
    records = []
    if download:
        image_directory.mkdir(parents=True, exist_ok=True)
    with index_path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            path = image_directory / Path(row["source_path"]).name
            if download and not path.is_file():
                with urlopen(DATA_URL + row["source_path"], timeout=30) as response:
                    data = response.read()
                if row["sha256"] != hashlib.sha256(data).hexdigest():
                    raise ValueError(f"Checksum mismatch after download: {path}")
                path.write_bytes(data)
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if row["sha256"] != digest:
                raise ValueError(f"Checksum mismatch: {path}")
            if digest in seen:
                continue
            seen.add(digest)
            records.append({**row, "path": path, "visible_damage": int(row["visible_damage"])})
    return records


def mobilenet_features(paths, boxes=None):
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    torch.set_num_threads(2)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    encoder = mobilenet_v3_small(weights=weights).eval()
    preprocess = weights.transforms()
    chunks = []
    with torch.inference_mode():
        for first in range(0, len(paths), 16):
            batch = []
            for index in range(first, min(first + 16, len(paths))):
                with Image.open(paths[index]) as photo:
                    image = photo.crop(boxes[index]) if boxes is not None else photo
                    batch.append(preprocess(ImageOps.exif_transpose(image).convert("RGB")))
            tensors = torch.stack(batch)
            chunks.append(encoder.avgpool(encoder.features(tensors)).flatten(1).numpy())
    return np.concatenate(chunks)


def evaluate(records, pretrained, synthetic_baseline=None):
    labels = np.array([row["visible_damage"] for row in records])
    groups = np.array([row["capture_group"] for row in records])
    paths = [row["path"] for row in records]
    hand_crafted = np.asarray([features(path) for path in paths])
    candidates = [(name, model, hand_crafted) for name, model in models().items()]
    if pretrained:
        candidates.append(("mobilenet_v3_small", models()["logistic"], mobilenet_features(paths)))

    folds = list(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42).split(
        hand_crafted, labels, groups
    ))
    predictions = [{"source_path": row["source_path"], "capture_group": row["capture_group"],
                    "visible_damage": row["visible_damage"]} for row in records]
    summary = {}
    for name, model, matrix in candidates:
        out_of_fold = np.empty(len(labels))
        fold_ap = []
        for fold_number, (train, test) in enumerate(folds, 1):
            if len(set(labels[train])) != 2:
                raise ValueError("Each training fold needs both classes.")
            model.fit(matrix[train], labels[train])
            scores = model.predict_proba(matrix[test])[:, 1]
            out_of_fold[test] = scores
            fold_ap.append(round(float(average_precision_score(labels[test], scores)), 4)
                           if len(set(labels[test])) == 2 else None)
            for idx, score in zip(test, scores, strict=True):
                predictions[idx]["fold"] = fold_number
                predictions[idx][name] = float(score)
        summary[name] = {
            "out_of_fold_ap": round(float(average_precision_score(labels, out_of_fold)), 4),
            "out_of_fold_roc_auc": round(float(roc_auc_score(labels, out_of_fold)), 4),
            "fold_ap": fold_ap,
        }
    if synthetic_baseline is not None:
        model = joblib.load(synthetic_baseline)["damage"]
        scores = model.predict_proba(hand_crafted)[:, 1]
        for prediction, score in zip(predictions, scores, strict=True):
            prediction["synthetic_baseline"] = float(score)
        summary["synthetic_baseline"] = {
            "ap": round(float(average_precision_score(labels, scores)), 4),
            "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        }
    return {
        "images": len(records), "capture_groups": len(set(groups)),
        "with_damage_annotation": int(labels.sum()), "positive_fraction": float(labels.mean()),
        "test_counts_by_fold": [{"fold": i, "images": len(test),
                                 "with_damage_annotation": int(labels[test].sum())}
                                for i, (_, test) in enumerate(folds, 1)],
        "models": summary,
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("real_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("run/real"))
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--synthetic-baseline", type=Path,
                        help="Optional locally generated model from parcel_cv.py demo.")
    parser.add_argument("--download", action="store_true", help="Fetch indexed files from the source repository.")
    args = parser.parse_args()
    try:
        records = load_photos(args.index, args.image_directory, download=args.download)
        report, predictions = evaluate(records, args.pretrained, args.synthetic_baseline)
    except (ImportError, OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
