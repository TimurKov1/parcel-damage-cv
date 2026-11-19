"""Generate reproducible mock claims with two photographs per parcel."""
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def render(rng, color, condition, defect, face):
    side = 128
    background = rng.choice([(80, 92, 104), (140, 154, 145), (185, 175, 165)])
    image = Image.new("RGB", (side, side), background)
    draw = ImageDraw.Draw(image)
    x, y = rng.randint(10, 23), rng.randint(22, 35)
    width, height = rng.randint(68, 87), rng.randint(48, 66)
    if condition == "cropped":
        x = -rng.randint(25, 40)
    front = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    draw.polygon([(x, y), (x + 13, y - 13), (x + width + 14, y - 13),
                  (x + width, y)], fill=tuple(min(c + 16, 255) for c in color))
    draw.polygon([(x + width, y), (x + width + 14, y - 13),
                  (x + width + 14, y + height - 12), (x + width, y + height)],
                 fill=tuple(c - 23 for c in color))
    draw.polygon(front, fill=color, outline=tuple(c - 37 for c in color))
    for _ in range(40):
        px, py = x + rng.randrange(width - 3), y + rng.randrange(height)
        shade = rng.randint(-15, 13)
        draw.line((px, py, px + rng.randint(1, 3), py),
                  fill=tuple(max(0, min(255, c + shade)) for c in color))
    tape = x + width // 2 + rng.randint(-8, 8)
    draw.rectangle((tape, y + 1, tape + 7, y + height - 1), fill=(195, 159, 107))
    if rng.random() < 0.8:
        label_x = x + (width - 30 if face == "back" else 9)
        draw.rectangle((label_x, y + 9, label_x + 20, y + 23), fill=(224, 220, 206))
        for offset in [3, 7, 11, 15]:
            draw.line((label_x + offset, y + 12, label_x + offset, y + 20), fill=(91, 88, 84))
    px, py = x + rng.randint(12, width - 14), y + rng.randint(27, height - 11)
    size = rng.randint(4, 12)
    if defect == "tear":
        draw.line([(px - size, py - 4), (px - 2, py + 4), (px + 3, py - 4),
                   (px + size, py + 2)], fill=(49, 37, 32), width=3)
    elif defect == "puncture":
        draw.polygon([(px, py - size), (px + size // 2, py),
                      (px + 1, py + size // 2), (px - size // 2, py + 1)], fill=(58, 43, 32))
    elif defect == "dent":
        cx, cy = x + width, y + height
        draw.polygon([(cx, cy), (cx - size * 2, cy), (cx - size, cy - size),
                      (cx, cy - size * 2)], fill=(119, 90, 65))
    if rng.random() < 0.25:
        sx, sy = x + rng.randrange(18, width - 10), y + 26
        draw.ellipse((sx, sy, sx + 8, sy + 5), fill=tuple(c - 35 for c in color))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.79, 1.18))
    if condition == "dark":
        image = ImageEnhance.Brightness(image).enhance(0.28)
    elif condition == "blur":
        image = image.filter(ImageFilter.GaussianBlur(4.3))
    return image


def generate(destination: Path, count: int, seed: int):
    rng = random.Random(seed)
    (destination / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    scenarios = ["none", "tear", "none", "dent", "none", "puncture"]
    for index in range(count):
        box_id = f"box-{index:04d}"
        defect = scenarios[index % len(scenarios)]
        damaged_face = rng.choice(["front", "back"])
        color = (rng.randint(156, 214), rng.randint(110, 148), rng.randint(65, 103))
        second_face = "front" if rng.random() < 0.14 else "back"
        for view_index, face in enumerate(["front", second_face]):
            condition = rng.choices(["clear", "dark", "blur", "cropped"], [77, 8, 8, 7])[0]
            view_defect = defect if face == damaged_face else "none"
            path = f"images/{box_id}-{view_index}.jpg"
            render(rng, color, condition, view_defect, face).save(destination / path, quality=87)
            rows.append({
                "path": path, "box_id": box_id, "claim_id": box_id, "face": face,
                "usable": int(condition == "clear"), "visible_damage": (
                    int(view_defect != "none") if condition == "clear" else ""
                ), "claim_damaged": int(defect != "none"), "defect": defect,
                "condition": condition,
            })
    with (destination / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
