"""Compare full-image, box-crop, texture and CNN models on grouped real data."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from parcel_cv import image_features, models
from real_benchmark import load_photos, mobilenet_features


TILE_SIZE = 96
TILE_COORDINATES = (0, 53, 107, 160)


def extract_inputs(records):
    import cv2

    hog = cv2.HOGDescriptor((128, 128), (16, 16), (8, 8), (8, 8), 9)
    arrays = {name: [] for name in ("full", "crop", "hog_full", "hog_crop", "local_crop", "pixels",
                                    "tile_features", "tile_labels", "background")}
    for row in records:
        with Image.open(row["path"]) as original:
            left, top, right, bottom = map(int, (row["left"], row["top"], row["right"], row["bottom"]))
            scaled = np.asarray(original.convert("RGB").resize((96, 96)))
            outside = np.ones((96, 96), dtype=bool)
            width, height = original.size
            x1, y1 = int(left * 96 / width), int(top * 96 / height)
            x2, y2 = int(np.ceil(right * 96 / width)), int(np.ceil(bottom * 96 / height))
            outside[max(0, y1 - 1):y2 + 1, max(0, x1 - 1):x2 + 1] = False
            background = scaled[outside]
            histogram = []
            for channel in range(3):
                counts = np.histogram(background[:, channel], bins=16, range=(0, 256))[0]
                histogram.extend(counts / max(1, len(background)))
            arrays["background"].append(histogram)
            crop = ImageOps.exif_transpose(original.crop((left, top, right, bottom))).convert("RGB")
            full = ImageOps.exif_transpose(original).convert("RGB")
        arrays["full"].append(image_features(full))
        arrays["crop"].append(image_features(crop))
        for key, image in (("hog_full", full), ("hog_crop", crop)):
            gray = np.asarray(image.convert("L").resize((128, 128)), dtype=np.uint8)
            arrays[key].append(hog.compute(gray).ravel())
        arrays["local_crop"].append(local_features(crop, cv2))
        arrays["pixels"].append(np.asarray(crop.resize((112, 112)), dtype=np.uint8))
        tile_features, tile_labels = extract_tiles(crop, json.loads(row["damage_boxes_relative"]))
        arrays["tile_features"].append(tile_features)
        arrays["tile_labels"].append(tile_labels)
    return {name: np.asarray(values) for name, values in arrays.items()}


def extract_tiles(crop, damage_boxes):
    size = 256
    resized = crop.resize((size, size))
    inputs, targets = [], []
    for y in TILE_COORDINATES:
        for x in TILE_COORDINATES:
            right, bottom = x + TILE_SIZE, y + TILE_SIZE
            inputs.append(image_features(resized.crop((x, y, right, bottom))))
            positive, touches = False, False
            for x1, y1, x2, y2 in np.asarray(damage_boxes) * size:
                overlap = (max(0, min(x2, right) - max(x1, x))
                           * max(0, min(y2, bottom) - max(y1, y)))
                touches |= overlap > 0
                positive |= (overlap >= 0.30 * (x2 - x1) * (y2 - y1)
                             and overlap >= 0.003 * TILE_SIZE ** 2)
            targets.append(1 if positive else (-1 if touches else 0))
    return inputs, targets


def tile_scores(estimator, arrays, train, test):
    tiles = arrays["tile_features"]
    labels = arrays["tile_labels"][train].ravel()
    training_tiles = tiles[train].reshape(len(labels), -1)
    eligible = labels >= 0
    model = clone(estimator).fit(training_tiles[eligible], labels[eligible])
    if len(model.classes_) != 2:
        raise ValueError("Tile model needs annotated positive and negative training tiles.")
    scores = model.predict_proba(tiles[test].reshape(-1, tiles.shape[-1]))[:, 1]
    scores = scores.reshape(len(test), -1)
    return scores.max(axis=1), scores.argmax(axis=1)


def local_features(crop, cv2):
    gray = np.asarray(crop.convert("L").resize((256, 256)), dtype=np.float32) / 255
    gradient = cv2.magnitude(cv2.Sobel(gray, -1, 1, 0), cv2.Sobel(gray, -1, 0, 1))
    laplacian = np.abs(cv2.Laplacian(gray, -1))
    output = []
    for width in (32, 64):
        for y in range(0, 256, width):
            for x in range(0, 256, width):
                region = np.s_[y:y + width, x:x + width]
                output.extend([gray[region].std(), gradient[region].mean(),
                               np.quantile(gradient[region], 0.95), laplacian[region].mean()])
    return output


def cnn_scores(pixels, labels, train, test, seed):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    tensors = torch.from_numpy(pixels).permute(0, 3, 1, 2).float() / 255
    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
        nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(), nn.Dropout(0.2), nn.Linear(48 * 9, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    target = torch.from_numpy(labels).float()
    positives = labels[train].sum()
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(train) - positives) / positives))
    rng = np.random.default_rng(seed)
    for _ in range(16):
        for batch in np.array_split(rng.permutation(train), int(np.ceil(len(train) / 16))):
            inputs = tensors[batch]
            flip = torch.rand(len(batch)) < 0.5
            inputs[flip] = inputs[flip].flip(-1)
            brightness = 0.85 + 0.30 * torch.rand(len(batch), 1, 1, 1)
            logits = model((inputs * brightness).clamp(0, 1)).squeeze(1)
            loss = loss_function(logits, target[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return model(tensors[test]).squeeze(1).sigmoid().numpy()


def metrics(labels, scores):
    if len(np.unique(labels)) != 2:
        raise ValueError("An evaluation partition needs both labels.")
    return {"ap": float(average_precision_score(labels, scores)),
            "roc_auc": float(roc_auc_score(labels, scores))}


def benchmark(records, pretrained=False):
    labels = np.array([row["visible_damage"] for row in records])
    groups = np.array([row["capture_group"] for row in records])
    partitions = np.array([row["partition"] for row in records])
    dev, test = np.flatnonzero(partitions == "development"), np.flatnonzero(partitions == "holdout")
    if len(dev) + len(test) != len(records) or set(groups[dev]) & set(groups[test]):
        raise ValueError("Every group must belong to exactly one partition.")
    arrays = extract_inputs(records)
    logistic, trees = (models()[key] for key in ("logistic", "extra_trees"))
    svm = make_pipeline(StandardScaler(), SVC(C=1, class_weight="balanced"))
    candidates = {
        "logistic_full": (logistic, "full"),
        "extra_trees_full": (trees, "full"),
        "extra_trees_crop": (trees, "crop"),
        "svm_hog_full": (svm, "hog_full"),
        "svm_hog_crop": (svm, "hog_crop"),
        "extra_trees_local_crop": (trees, "local_crop"),
        "supervised_tiles_crop": (trees, "tile_features"),
        "cnn_crop": (None, "pixels"),
    }
    fold_splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    folds = [(dev[train], dev[held_out]) for train, held_out in
             fold_splitter.split(dev, labels[dev], groups[dev])]
    predicted = [{"source_path": row["source_path"], "capture_group": row["capture_group"],
                  "visible_damage": row["visible_damage"], "partition": row["partition"],
                  "fold": ""} for row in records]
    for fold, (_, rows) in enumerate(folds, 1):
        for index in rows:
            predicted[index]["fold"] = fold
    summary = {}
    top_tile_hits = None
    if pretrained:
        paths = [row["path"] for row in records]
        boxes = [tuple(int(row[key]) for key in ("left", "top", "right", "bottom")) for row in records]
        arrays["mobilenet_full"] = mobilenet_features(paths)
        arrays["mobilenet_crop"] = mobilenet_features(paths, boxes)
        for key in ("mobilenet_full", "mobilenet_crop"):
            candidates[key] = (logistic, key)
    background_control = "logistic_background_only"
    candidates[background_control] = (logistic, "background")
    for name, (model, key) in candidates.items():
        fold_reports = []
        for fold, (train, held_out) in enumerate(folds + [(dev, test)], 1):
            if key == "tile_features":
                scores, top_tiles = tile_scores(model, arrays, train, held_out)
                for index, tile in zip(held_out, top_tiles, strict=True):
                    predicted[index]["best_tile"] = int(tile)
                if fold == len(folds) + 1:
                    top_tile_hits = int((arrays["tile_labels"][held_out, top_tiles] == 1).sum())
            elif model is None:
                scores = cnn_scores(arrays[key], labels, train, held_out, seed=42 + fold)
            else:
                estimator = clone(model).fit(arrays[key][train], labels[train])
                if hasattr(estimator, "predict_proba"):
                    scores = estimator.predict_proba(arrays[key][held_out])[:, 1]
                else:
                    scores = estimator.decision_function(arrays[key][held_out])
            for index, score in zip(held_out, scores, strict=True):
                predicted[index][name] = float(score)
            fold_reports.append(metrics(labels[held_out], scores))
        summary[name] = {
            "development_folds": fold_reports[:-1],
            "development_mean_ap": float(np.mean([fold["ap"] for fold in fold_reports[:-1]])),
            "holdout": fold_reports[-1],
        }
    selected = max((name for name in summary if name != background_control),
                   key=lambda name: summary[name]["development_mean_ap"])
    size_threshold = 0.03
    small = [i for i in test if labels[i] == 1 and
             float(records[i]["largest_defect_fraction"]) < size_threshold]
    other = [i for i in test if labels[i] == 1 and i not in small]
    for name, result in summary.items():
        # Inspect retrieval when reviewers can read only the highest-scoring half.
        inspected = set(sorted(test, key=lambda i: predicted[i][name], reverse=True)[:len(test) // 2])
        result["review_top_half"] = {
            "reviewed": len(inspected),
            "small_defects_found": sum(i in inspected for i in small),
            "other_defects_found": sum(i in inspected for i in other),
        }
    rng = np.random.default_rng(43)
    group_rows = [test[groups[test] == group] for group in sorted(set(groups[test]))]
    selected_scores = np.array([row[selected] for row in predicted])
    bootstrap_ap = []
    for drawn_groups in rng.integers(0, len(group_rows), size=(1000, len(group_rows))):
        sample = np.concatenate([group_rows[group] for group in drawn_groups])
        if len(np.unique(labels[sample])) == 2:
            bootstrap_ap.append(average_precision_score(labels[sample], selected_scores[sample]))
    return {
        "development": {"images": len(dev), "groups": len(set(groups[dev])), "positive": int(labels[dev].sum())},
        "holdout": {"images": len(test), "groups": len(set(groups[test])), "positive": int(labels[test].sum())},
        "size_slices": {"small_defect_area_fraction_below": size_threshold,
                        "small_defects": len(small), "other_defects": len(other)},
        "selected_on_development": selected,
        "selected_holdout_ap_group_bootstrap_95": np.quantile(bootstrap_ap, [0.025, 0.975]).tolist(),
        "tile_localization": {"top_tile_matches_defect": top_tile_hits,
                              "positive_holdout_photos": int(labels[test].sum())},
        "models": summary,
    }, predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--index", type=Path, default=Path(__file__).with_name("real_expanded_index.csv"))
    parser.add_argument("--out", type=Path, default=Path("run/real-expanded"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()
    try:
        records = load_photos(args.index, args.image_directory, download=args.download)
        report, predictions = benchmark(records, pretrained=args.pretrained)
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
