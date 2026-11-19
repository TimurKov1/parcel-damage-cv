"""Train, evaluate and run a parcel photograph triage prototype."""
import argparse
import csv
import itertools
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from synthetic import generate


FACES = {"front", "back"}
REVIEW_ROUTES = {"more_photos", "manual_review"}
MISSED_DAMAGE_COST = 5
FALSE_DAMAGE_COST = 3
REVIEW_COST = 1
LIMITS = [(0.10, 0.90), (0.20, 0.80), (0.30, 0.70), (0.40, 0.60), (0.50, 0.50)]


def features(path):
    with Image.open(path) as image:
        return image_features(ImageOps.exif_transpose(image))


def image_features(image):
    image = image.convert("RGB")
    gray = np.asarray(image.convert("L").resize((64, 64)), dtype=float) / 255
    rgb = np.asarray(image.resize((48, 48)), dtype=float) / 255
    values = [gray.mean(), gray.std()]
    # Color statistics and edge histograms retain coarse spatial information.
    for row in range(4):
        for col in range(4):
            cell = gray[16 * row:16 * (row + 1), 16 * col:16 * (col + 1)]
            dy, dx = np.gradient(cell)
            strength = np.hypot(dx, dy)
            direction = np.arctan2(dy, dx) % np.pi
            hist, _ = np.histogram(direction, bins=8, range=(0, np.pi), weights=strength)
            color = rgb[12 * row:12 * (row + 1), 12 * col:12 * (col + 1)]
            values.extend([cell.std(), strength.mean(), (strength > 0.10).mean()])
            values.extend(hist / cell.size)
            values.extend(color.mean(axis=(0, 1)))
    return values


def models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500)),
        "extra_trees": ExtraTreesClassifier(n_estimators=160, min_samples_leaf=3,
                                            class_weight="balanced", random_state=42, n_jobs=1),
    }


def load_manifest(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The manifest is empty.")
    claims = {}
    for row in rows:
        if row["face"] not in FACES:
            raise ValueError("Every photo needs a declared face: front or back.")
        if row["usable"] not in {"0", "1"} or row["claim_damaged"] not in {"0", "1"}:
            raise ValueError("usable and claim_damaged must be 0 or 1.")
        expected = {"0", "1"} if row["usable"] == "1" else {""}
        if row["visible_damage"] not in expected:
            raise ValueError("Set visible_damage to 0 or 1 for usable views, otherwise leave it blank.")
        labels = (row["box_id"], row["claim_damaged"], row["defect"])
        if claims.setdefault(row["claim_id"], labels) != labels:
            raise ValueError("All photos in a claim must share the box and claim labels.")
    return rows


def split_rows(rows):
    groups = np.array([row["box_id"] for row in rows])
    train_val, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42).split(groups, groups=groups))
    train, val = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=43).split(train_val, groups=groups[train_val]))
    return train_val[train], train_val[val], test


def score_images(rows, quality, damage):
    return [
        {**row, "quality_score": float(q), "damage_score": float(d)}
        for row, q, d in zip(rows, quality, damage, strict=True)
    ]


def triage(photos, low, high):
    usable = [photo for photo in photos if photo["quality_score"] >= 0.5]
    score = max((photo["damage_score"] for photo in usable), default=None)
    if score is not None and score >= high:
        return "visible_damage", score
    if {photo["face"] for photo in usable} != FACES:
        return "more_photos", score
    if score is not None and score < low:
        return "no_visible_damage", score
    return "manual_review", score


def route_claims(photos, low, high, first_only=False):
    claims = []
    by_claim = sorted(photos, key=lambda row: row["claim_id"])
    for claim_id, entries in itertools.groupby(by_claim, key=lambda row: row["claim_id"]):
        views = list(entries)
        selected = views[:1] if first_only else views
        route, score = triage(selected, low, high)
        claims.append({
            "claim_id": claim_id, "damaged": int(views[0]["claim_damaged"]),
            "defect": views[0]["defect"], "photos": len(selected), "route": route, "score": score,
        })
    return claims


def summarize(claims):
    reviewed = sum(row["route"] in REVIEW_ROUTES for row in claims)
    return {
        "claims": len(claims),
        "damaged_claims": sum(row["damaged"] for row in claims),
        "review_or_more_photos": reviewed,
        "automatic_coverage": round(1 - reviewed / len(claims), 4),
        "detected_damage": sum(row["damaged"] == 1 and row["route"] == "visible_damage"
                               for row in claims),
        "missed_damage": sum(row["damaged"] == 1 and row["route"] == "no_visible_damage"
                             for row in claims),
        "false_damage": sum(row["damaged"] == 0 and row["route"] == "visible_damage"
                            for row in claims),
    }


def ap(labels, scores):
    return round(float(average_precision_score(labels, scores)), 4) if len(set(labels)) == 2 else None


def evaluate(manifest, output, source):
    rows = load_manifest(manifest)
    train, validation, test = split_rows(rows)
    matrix = np.array([features(manifest.parent / row["path"]) for row in rows])
    usable_train = [i for i in train if rows[i]["usable"] == "1"]
    quality_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))
    quality_model.fit(matrix[train], [int(rows[i]["usable"]) for i in train])
    quality_scores = quality_model.predict_proba(matrix)[:, 1]
    results, trials = {}, []
    for name, model in models().items():
        model.fit(matrix[usable_train], [int(rows[i]["visible_damage"]) for i in usable_train])
        damage_scores = model.predict_proba(matrix)[:, 1]
        validation_photos = score_images([rows[i] for i in validation],
                                         quality_scores[validation], damage_scores[validation])
        test_photos = score_images([rows[i] for i in test], quality_scores[test], damage_scores[test])
        curve = []
        for low, high in LIMITS:
            summary = summarize(route_claims(validation_photos, low, high))
            curve.append({"low": low, "high": high, **summary})
            cost = (MISSED_DAMAGE_COST * summary["missed_damage"]
                    + FALSE_DAMAGE_COST * summary["false_damage"]
                    + REVIEW_COST * summary["review_or_more_photos"])
            trials.append((cost, summary["missed_damage"], summary["false_damage"],
                           -summary["automatic_coverage"], name, low, high))
        visible = [photo for photo in test_photos if photo["usable"] == "1"]
        results[name] = {"model": model, "test_photos": test_photos, "validation_curve": curve,
                         "damage_photo_ap": ap([int(p["visible_damage"]) for p in visible],
                                               [p["damage_score"] for p in visible])}
    cost, _, _, _, selected, low, high = min(trials)
    chosen = results[selected]
    test_claims = route_claims(chosen["test_photos"], low, high)
    report = {
        "source": source, "photos": len(rows),
        "boxes_by_split": dict(zip(["train", "validation", "test"],
                                   [len({rows[i]["box_id"] for i in indexes})
                                    for indexes in (train, validation, test)])),
        "quality_photo_ap": ap([int(rows[i]["usable"]) for i in test], quality_scores[test]),
        "model_comparison": {name: {k: record[k] for k in ("damage_photo_ap", "validation_curve")}
                             for name, record in results.items()},
        "selected_model": selected, "low": low, "high": high,
        "validation_cost": cost,
        "cost_weights": {"missed_damage": MISSED_DAMAGE_COST, "false_damage": FALSE_DAMAGE_COST,
                         "review_or_more_photos": REVIEW_COST},
        "test_all_photos": summarize(test_claims),
        "test_first_photo": summarize(route_claims(chosen["test_photos"], low, high, first_only=True)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "test_claims.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(test_claims[0]))
        writer.writeheader()
        writer.writerows(test_claims)
    (output / "split.json").write_text(json.dumps({
        name: sorted({rows[i]["box_id"] for i in indexes})
        for name, indexes in zip(("train", "validation", "test"), (train, validation, test))
    }, indent=2) + "\n", encoding="utf-8")
    with sqlite3.connect(output / "analysis.sqlite") as db:
        with db:
            db.execute("DROP TABLE IF EXISTS test_claims")
            db.execute("""CREATE TABLE test_claims (
                claim_id TEXT PRIMARY KEY, damaged INTEGER, defect TEXT,
                photos INTEGER, route TEXT, score REAL)""")
            db.executemany("INSERT INTO test_claims VALUES (:claim_id, :damaged, :defect, :photos, :route, :score)",
                           test_claims)
        sql = Path(__file__).with_name("analysis.sql").read_text(encoding="utf-8")
        cursor = db.execute(sql)
        with (output / "errors_by_defect.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([column[0] for column in cursor.description])
            writer.writerows(cursor)
    joblib.dump({"quality": quality_model, "damage": chosen["model"], "low": low, "high": high},
                output / "model.joblib")
    return report


def predict_photos(bundle, views):
    matrix = [features(source) for _, source in views]
    photos = score_images([{"face": face} for face, _ in views],
                          bundle["quality"].predict_proba(matrix)[:, 1],
                          bundle["damage"].predict_proba(matrix)[:, 1])
    route, score = triage(photos, bundle["low"], bundle["high"])
    return {"route": route, "damage_score": score, "photos": photos}


def predict(model_path, arguments):
    views = []
    for argument in arguments:
        face, separator, name = argument.partition("=")
        if face not in FACES or not separator or not name:
            raise ValueError("Use front=path.jpg or back=path.jpg.")
        views.append((face, Path(name)))
    result = predict_photos(joblib.load(model_path), views)
    for photo, (_, path) in zip(result["photos"], views, strict=True):
        photo["path"] = str(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo")
    demo.add_argument("--count", type=int, default=150)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--data", type=Path, default=Path("data/demo"))
    demo.add_argument("--out", type=Path, default=Path("run/demo"))
    train = commands.add_parser("train")
    train.add_argument("manifest", type=Path)
    train.add_argument("--out", type=Path, default=Path("run/custom"))
    infer = commands.add_parser("predict")
    infer.add_argument("model", type=Path)
    infer.add_argument("photos", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            if args.count < 30:
                parser.error("demo requires at least 30 claims")
            generate(args.data, args.count, args.seed)
            result = evaluate(args.data / "manifest.csv", args.out, "synthetic")
        elif args.command == "train":
            result = evaluate(args.manifest, args.out, "custom")
        else:
            result = predict(args.model, args.photos)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Ошибка: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
