"""Local demonstration for the synthetic parcel model."""
import csv
import io
import json
from pathlib import Path

import joblib
import streamlit as st

from parcel_cv import predict_photos


ROUTES = {
    "visible_damage": "Видно повреждение",
    "no_visible_damage": "На снимках повреждений не видно",
    "more_photos": "Нужны дополнительные фотографии",
    "manual_review": "Нужна проверка оператором",
}
EXAMPLES = {
    "Разрыв коробки": "box-0073",
    "Повреждение видно на втором снимке": "box-0143",
    "Недостаточно ракурсов": "box-0018",
    "Ошибка модели — пропущенная вмятина": "box-0045",
}
ROOT = Path(__file__).resolve().parent
RUN = ROOT / "run" / "demo"
DATA = ROOT / "data" / "demo"

st.set_page_config(page_title="Проверка фото посылки", layout="wide")
st.title("Проверка фото посылки")
st.caption("Учебный прототип. Модель обучена на синтетических изображениях.")

if not (RUN / "model.joblib").is_file():
    st.error("Сначала запусти из папки проекта: python parcel_cv.py demo")
    st.stop()


@st.cache_resource
def load_model(path, modified_at):
    return joblib.load(path)


model_path = RUN / "model.joblib"
bundle = load_model(model_path, model_path.stat().st_mtime_ns)
mode = st.radio("Фото для проверки", ("Пример", "Мои фото"), horizontal=True)
views = []
if mode == "Пример":
    example = st.selectbox("Обращение", list(EXAMPLES))
    with (DATA / "manifest.csv").open(newline="", encoding="utf-8") as file:
        photos = [row for row in csv.DictReader(file) if row["claim_id"] == EXAMPLES[example]]
    if not photos:
        st.info("Примеры доступны после запуска генератора с параметрами по умолчанию.")
    use_second = st.checkbox("Показать второй снимок модели", value=True)
    selected = photos if use_second else photos[:1]
    views = [(row["face"], (DATA / row["path"]).read_bytes()) for row in selected]
else:
    for face, label in (("front", "Передняя сторона"), ("back", "Задняя сторона")):
        photo = st.file_uploader(label, type=["jpg", "jpeg", "png"])
        if photo is not None:
            views.append((face, photo.getvalue()))

if views:
    try:
        result = predict_photos(bundle, [(face, io.BytesIO(content)) for face, content in views])
    except (OSError, ValueError) as error:
        st.error(f"Не удалось обработать фото. {error}")
    else:
        st.subheader(ROUTES[result["route"]])
        for column, (face, content), photo in zip(
            st.columns(len(views)), views, result["photos"], strict=True
        ):
            with column:
                label = "Передняя сторона" if face == "front" else "Задняя сторона"
                st.image(content, caption=label, width=240)
                st.caption(f"Оценка пригодности {photo['quality_score']:.2f} · "
                           f"оценка повреждения {photo['damage_score']:.2f}")
        if mode == "Пример" and example == "Ошибка модели — пропущенная вмятина":
            st.warning("В генераторе для этого случая задана вмятина.")

with st.expander("Результаты на отложенной выборке"):
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    test = metrics["test_all_photos"]
    cols = st.columns(3)
    cols[0].metric("Автоматическая сортировка", f"{test['automatic_coverage']:.1%}")
    cols[1].metric("Обращений в тесте", test["claims"])
    cols[2].metric("Пропущено повреждений", test["missed_damage"])
    st.caption("Результаты относятся к синтетическим изображениям.")

    st.markdown("**Компромисс на валидации**")
    rows = []
    for point in metrics["model_comparison"][metrics["selected_model"]]["validation_curve"]:
        rows.append({
            "На проверку или пересъёмку, %": round(100 * point["review_or_more_photos"]
                                                  / point["claims"], 1),
            "Пропущено повреждений": point["missed_damage"],
            "Ложно найдено повреждений": point["false_damage"],
            "Пороги": f"{point['low']:.1f} / {point['high']:.1f}",
        })
    st.scatter_chart(rows, x="На проверку или пересъёмку, %",
                     y=["Пропущено повреждений", "Ложно найдено повреждений"])
    st.dataframe(rows, hide_index=True)
