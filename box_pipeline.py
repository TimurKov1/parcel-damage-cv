"""Train and evaluate automatic parcel cropping and image-level damage ranking."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from parcel_cv import models
from real_benchmark import load_photos, mobilenet_features


SIDE = 112
GAP = 3
COORDINATES = np.indices((SIDE, SIDE), dtype=np.float32) / (SIDE - 1)
HOG = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)


def read_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def make_grid(image):
    rgb = np.asarray(image.resize((SIDE, SIDE), Image.Resampling.BILINEAR))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    smoothed = cv2.GaussianBlur(rgb, (7, 7), 0)
    features = np.concatenate((rgb / 255, hsv / 255, smoothed / 255, COORDINATES.transpose(1, 2, 0)),
                              axis=-1).reshape(-1, 11).astype(np.float32)
    return hsv, features


def largest_box(mask):
    """Return coordinates in the normalized grid, with a little crop padding."""
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
    _, _, stats, _ = cv2.connectedComponentsWithStats(opened)
    if len(stats) == 1:
        return None
    left, top, width, height, area = max(stats[1:], key=lambda row: row[cv2.CC_STAT_AREA])
    if area < SIDE ** 2 * 0.035:
        return None
    return (max(0, int(left) - GAP), max(0, int(top) - GAP),
            min(SIDE, int(left + width) + GAP), min(SIDE, int(top + height) + GAP))


def color_box(hsv):
    hue, saturation, value = hsv.transpose(2, 0, 1)
    return largest_box((hue >= 4) & (hue <= 38) & (saturation >= 38) & (value >= 55))


def learned_box(detector, pixel_features):
    probability = detector.predict_proba(pixel_features)[:, 1]
    return largest_box(probability.reshape(SIDE, SIDE) >= 0.50)


def fit_detector(inputs, boxes, indices):
    rng = np.random.default_rng(42)
    features, targets = [], []
    y, x = COORDINATES.reshape(2, -1)
    for index in indices:
        left, top, right, bottom = boxes[index] / SIDE
        positive = (left <= x) & (x < right) & (top <= y) & (y < bottom)
        picked = rng.choice(len(positive), size=500, replace=False)
        features.append(inputs[index][picked])
        targets.append(positive[picked])
    return ExtraTreesClassifier(n_estimators=64, max_depth=14, min_samples_leaf=12,
                                class_weight="balanced", random_state=42, n_jobs=2).fit(
                                    np.concatenate(features), np.concatenate(targets)
                                )


def native_box(size, box):
    width, height = size
    if box is None:
        return (0, 0, width, height)
    left, top, right, bottom = box
    return (int(left * width / SIDE), int(top * height / SIDE),
            int(np.ceil(right * width / SIDE)), int(np.ceil(bottom * height / SIDE)))


def crop_grid(image, box):
    return image.crop(native_box(image.size, box))


def hog(image):
    normalized = ImageOps.exif_transpose(image).convert("L").resize((128, 128))
    return HOG.compute(np.asarray(normalized, dtype=np.uint8)).ravel()


def classifier():
    return make_pipeline(StandardScaler(), SVC(C=1, class_weight="balanced"))


def intersection_over_union(predicted, truth):
    if predicted is None:
        return 0.0
    left, top = np.maximum(predicted[:2], truth[:2])
    right, bottom = np.minimum(predicted[2:], truth[2:])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = (predicted[2] - predicted[0]) * (predicted[3] - predicted[1])
    area_b = (truth[2] - truth[0]) * (truth[3] - truth[1])
    return float(intersection / (area_a + area_b - intersection))


def evaluate(records, bundle_path, pretrained=False):
    y = np.array([r["visible_damage"] for r in records])
    groups = np.array([r["capture_group"] for r in records])
    dev = np.array([i for i, row in enumerate(records) if row["partition"] == "development"])
    test = np.array([i for i, row in enumerate(records) if row["partition"] == "holdout"])
    if len(test) + len(dev) != len(records) or set(groups[dev]) & set(groups[test]):
        raise ValueError("Each capture group must have one valid partition.")
    inputs, refs, whole, oracle, heuristic, shifted_crops = [], [], [], [], [], []
    rng = np.random.default_rng(43)
    sizes, manual_rectangles = [], []
    for row in records:
        image = read_image(row["path"])
        left, top, right, bottom = (int(row[k]) for k in ("left", "top", "right", "bottom"))
        width, height = image.size
        sizes.append(image.size)
        manual_rectangles.append((left, top, right, bottom))
        refs.append(np.array([left / width, top / height, right / width, bottom / height]) * SIDE)
        hsv, pixels = make_grid(image)
        inputs.append(pixels)
        heuristic.append(color_box(hsv))
        whole.append(hog(image))
        exact_box = np.array([left, top, right, bottom])
        oracle.append(hog(image.crop(exact_box)))
        box_size = np.array([right - left, bottom - top] * 2)
        shifts = rng.uniform(-0.16, 0.16, size=(3, 4)) * box_size
        variants = np.clip(exact_box + shifts, 0, [width, height, width, height]).astype(int)
        shifted_crops.append([hog(image.crop(variant)) for variant in variants])
    whole, oracle, shifted_crops = map(np.asarray, (whole, oracle, shifted_crops))
    methods = ("full", "manual_box", "color_box", "learned_box", "learned_box_jitter")
    paths = [row["path"] for row in records]
    if pretrained:
        mobile_features = mobilenet_features(paths, manual_rectangles)
        methods += ("mobilenet_color_box", "mobilenet_learned_box")
    folds = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    parts = [(dev[a], dev[b]) for a, b in folds.split(dev, y[dev], groups[dev])] + [(dev, test)]
    reports = {name: [] for name in methods}
    predictions = []
    for fold, (train, heldout) in enumerate(parts):
        detector = fit_detector(inputs, refs, train)
        whole_classifier = classifier().fit(whole[train], y[train])
        crop_classifier = classifier().fit(oracle[train], y[train])
        augmented = np.concatenate((oracle[train, None, :], shifted_crops[train]), axis=1)
        augmented_classifier = classifier().fit(
            augmented.reshape(-1, oracle.shape[-1]), np.repeat(y[train], augmented.shape[1])
        )
        learned = [learned_box(detector, inputs[i]) for i in heldout]
        boxes = {"color_box": [heuristic[i] for i in heldout], "learned_box": learned, "learned_box_jitter": learned}
        scores = {"full": whole_classifier.decision_function(whole[heldout]),
                  "manual_box": crop_classifier.decision_function(oracle[heldout])}
        for name, guesses in boxes.items():
            crops = [hog(crop_grid(read_image(records[i]["path"]), guess))
                     if guess is not None else whole[i]
                     for i, guess in zip(heldout, guesses, strict=True)]
            fitted = augmented_classifier if name == "learned_box_jitter" else crop_classifier
            scores[name] = fitted.decision_function(crops)
        if pretrained:
            mobile_head = models()["logistic"].fit(mobile_features[train], y[train])
            for method in ("color_box", "learned_box"):
                key = "mobilenet_" + method
                boxes[key] = boxes[method]
                coordinates = [native_box(sizes[i], box)
                               for i, box in zip(heldout, boxes[method], strict=True)]
                embeddings = mobilenet_features([paths[i] for i in heldout], coordinates)
                scores[key] = mobile_head.decision_function(embeddings)
        for name in methods:
            report = {"ap": float(average_precision_score(y[heldout], scores[name])),
                      "roc_auc": float(roc_auc_score(y[heldout], scores[name]))}
            if name in boxes:
                ious = [intersection_over_union(guess, refs[i])
                        for i, guess in zip(heldout, boxes[name], strict=True)]
                report.update(mean_iou=float(np.mean(ious)),
                              iou_at_least_50=sum(iou >= 0.5 for iou in ious),
                              fallback_to_full=sum(guess is None for guess in boxes[name]))
            reports[name].append(report)
        if fold == len(parts) - 1:
            for j, index in enumerate(heldout):
                predictions.append({
                    "source_path": records[index]["source_path"], "visible_damage": int(y[index]),
                    **{name + "_score": float(scores[name][j]) for name in methods},
                    **{name + "_iou": intersection_over_union(guesses[j], refs[index])
                       for name, guesses in boxes.items()},
                })
            candidates = ("learned_box", "learned_box_jitter")
            selected = max(candidates, key=lambda name: np.mean([r["ap"] for r in reports[name][:-1]]))
            final_classifier = augmented_classifier if selected == "learned_box_jitter" else crop_classifier
            bundle = {"detector": detector, "classifier": final_classifier}
            if pretrained:
                bundle["mobilenet_head"] = mobile_head
            joblib.dump(bundle, bundle_path, compress=3)
    summaries = {name: {"development_mean_ap": float(np.mean([r["ap"] for r in folds[:-1]])),
                        "holdout": folds[-1]} for name, folds in reports.items()}
    for name in methods:
        if "mean_iou" not in reports[name][-1]:
            continue
        summaries[name]["development_mean_iou"] = float(
            np.mean([r["mean_iou"] for r in reports[name][:-1]])
        )
    return {"development_images": len(dev), "holdout_images": len(test),
            "holdout_positive": int(y[test].sum()), "selected_learned_variant": selected,
            "models": summaries}, predictions


def predict(model_path, image_path, method="hog"):
    bundle = joblib.load(model_path)
    image = read_image(image_path)
    _, features = make_grid(image)
    box = learned_box(bundle["detector"], features)
    if method == "mobilenet":
        if "mobilenet_head" not in bundle:
            raise ValueError("Rebuild the model with evaluate --pretrained.")
        embeddings = mobilenet_features([image_path], [native_box(image.size, box)])
        score = bundle["mobilenet_head"].decision_function(embeddings)[0]
    else:
        score = bundle["classifier"].decision_function([hog(crop_grid(image, box))])[0]
    return {"box_xyxy_relative": [position / SIDE for position in box] if box else None,
            "damage_ranking_score": float(score), "used_full_image": box is None,
            "classifier": method}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("evaluate")
    fit.add_argument("image_directory", type=Path)
    fit.add_argument("--download", action="store_true")
    fit.add_argument("--pretrained", action="store_true")
    fit.add_argument("--out", type=Path, default=Path("run/box-pipeline"))
    infer = commands.add_parser("predict")
    infer.add_argument("model", type=Path)
    infer.add_argument("image", type=Path)
    infer.add_argument("--method", choices=("hog", "mobilenet"), default="hog")
    args = parser.parse_args()
    if args.command == "predict":
        result = predict(args.model, args.image, args.method)
    else:
        index = Path(__file__).with_name("real_expanded_index.csv")
        records = load_photos(index, args.image_directory, download=args.download)
        args.out.mkdir(parents=True, exist_ok=True)
        result, rows = evaluate(records, args.out / "model.joblib", pretrained=args.pretrained)
        (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        with (args.out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
