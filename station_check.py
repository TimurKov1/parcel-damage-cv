"""Score a small manually reviewed courier-station sample with the frozen model."""
import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

from box_pipeline import color_box, crop_grid, hog, learned_box, make_grid, native_box, read_image
from real_benchmark import load_photos, mobilenet_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("run/station"))
    args = parser.parse_args()
    records = load_photos(Path(__file__).with_name("station_index.csv"),
                          args.image_directory, download=args.download)
    bundle = joblib.load(args.model)
    predictions, learned_coordinates = [], []
    for row in records:
        image = read_image(row["path"])
        hsv, features = make_grid(image)
        result = {"source_path": row["source_path"], "visible_damage": row["visible_damage"]}
        boxes = {"full": None, "color_box": color_box(hsv),
                 "learned_box": learned_box(bundle["detector"], features)}
        learned_coordinates.append(native_box(image.size, boxes["learned_box"]))
        for method, box in boxes.items():
            score = bundle["classifier"].decision_function([hog(crop_grid(image, box))])[0]
            result[method + "_score"] = float(score)
        predictions.append(result)
    methods = ("full", "color_box", "learned_box")
    if "mobilenet_head" in bundle:
        features = mobilenet_features([row["path"] for row in records], learned_coordinates)
        for row, score in zip(predictions, bundle["mobilenet_head"].decision_function(features), strict=True):
            row["mobilenet_learned_box_score"] = float(score)
        methods += ("mobilenet_learned_box",)
    labels = [row["visible_damage"] for row in predictions]
    report = {"images": len(predictions), "positive": sum(labels), "methods": {}}
    for method in methods:
        scores = [row[method + "_score"] for row in predictions]
        report["methods"][method] = {"ap": float(average_precision_score(labels, scores)),
                                     "roc_auc": float(roc_auc_score(labels, scores))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
