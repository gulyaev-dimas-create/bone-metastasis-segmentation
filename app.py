import streamlit as st
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import pydicom
import matplotlib.pyplot as plt
from io import BytesIO
import os
import gdown

st.set_page_config(
    page_title="Сегментация костных метастазов",
    page_icon="🏥",
    layout="wide"
)

CROP_TOP, CROP_BOTTOM = 30, 50
CROP_LEFT, CROP_RIGHT = 35, 570
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PATCH_SIZE = 128
STRIDE = 64

ZONE_BOUNDARIES = [
    ('Череп', 0, 100),
    ('Грудная клетка', 100, 300),
    ('Таз', 300, 440),
    ('Нижние конечности', 440, None)
]

MODEL_DRIVE_ID = "15vdWpMf86ylUgRV6TiW8xMtGd1cqEyuI"

@st.cache_resource
def load_model():
    model_path = "unet_best.pth"
    if not os.path.exists(model_path):
        # Скачиваем без гифок, просто текст
        st.info("Скачиваю модель с Google Диска (≈100 МБ)... подождите пару минут.")
        url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
        gdown.download(url, model_path, quiet=False)
        st.success("Модель загружена.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.UnetPlusPlus(
        encoder_name='resnet34',
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def remove_contour(dicom_bytes):
    dcm = pydicom.dcmread(BytesIO(dicom_bytes))
    img = dcm.pixel_array
    if len(img.shape) == 3:
        color_img = img.copy()
        red_channel = img[:, :, 0]
    else:
        color_img = np.stack([img, img, img], axis=2)
        red_channel = img
    mask_contour = (red_channel < 50).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    mask_contour = cv2.dilate(mask_contour, kernel, iterations=2)
    return cv2.inpaint(color_img, mask_contour, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

def remove_linear_artifacts(mask, max_aspect_ratio=6):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if min(w, h) > 0 and max(w, h) / min(w, h) < max_aspect_ratio:
            cleaned_mask[labels == i] = 255
    return cleaned_mask

def predict_sliding_window(image, model, device):
    h, w = image.shape[:2]
    prediction_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    total_y = (h - PATCH_SIZE) // STRIDE + 1
    total_x = (w - PATCH_SIZE) // STRIDE + 1
    total = total_y * total_x
    done = 0

    # Только текст, без анимаций
    status_text = st.empty()
    status_text.text(f"Инференс: 0/{total} патчей")

    for y in range(0, h - PATCH_SIZE + 1, STRIDE):
        for x in range(0, w - PATCH_SIZE + 1, STRIDE):
            patch = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE].astype(np.float32) / 255.0
            patch = (patch - MEAN) / STD
            patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                prob = torch.sigmoid(model(patch_tensor)).cpu().numpy()[0,0]
            prediction_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += prob
            count_map[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
            done += 1
            if done % 100 == 0:
                status_text.text(f"Инференс: {done}/{total} патчей")

    status_text.empty()
    prediction_map = np.divide(prediction_map, count_map, where=count_map>0)
    return prediction_map

def calculate_bsi(pred_mask, h, w):
    results = []
    total_pixels = 0
    total_meta = 0
    for zone_name, y_start, y_end in ZONE_BOUNDARIES:
        if y_end is None:
            y_end = h
        zone_pixels = (y_end - y_start) * w
        meta_in_zone = np.sum(pred_mask[y_start:y_end, :] > 0)
        bsi = (meta_in_zone / zone_pixels) * 100 if zone_pixels > 0 else 0
        results.append({
            'zone': zone_name,
            'pixels': int(meta_in_zone),
            'bsi': round(bsi, 2)
        })
        total_pixels += zone_pixels
        total_meta += meta_in_zone
    total_bsi = round((total_meta / total_pixels) * 100, 2) if total_pixels > 0 else 0
    return results, total_bsi

def create_overlay(image, pred_mask):
    h, w = image.shape[:2]
    overlay = image.copy()
    mask_3d = np.stack([pred_mask / 255.0, np.zeros_like(pred_mask), np.zeros_like(pred_mask)], axis=2)
    alpha = 0.4
    overlay = (overlay * (1 - alpha) + mask_3d * 255 * alpha).astype(np.uint8)
    for name, y_start, y_end in ZONE_BOUNDARIES:
        if y_end is None:
            y_end = h
        cv2.line(overlay, (0, y_start), (w, y_start), (255, 255, 255), 2)
        if y_end < h:
            cv2.line(overlay, (0, y_end), (w, y_end), (255, 255, 255), 2)
        y_mid = (y_start + y_end) // 2
        cv2.putText(overlay, name, (5, y_mid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return overlay

# ========================= ИНТЕРФЕЙС =========================
st.title("🏥 Сегментация костных метастазов")
st.markdown("### Автоматический анализ сцинтиграфических снимков")
st.markdown("*КрасГМУ им. В.Ф. Войно-Ясенецкого • КККОД им. А.И. Крыжановского*")

with st.sidebar:
    st.header("⚙️ Настройки анализа")
    threshold = st.slider("Порог бинаризации", 0.1, 0.9, 0.3, 0.05)
    st.subheader("Постобработка")
    use_artifact_removal = st.checkbox("Удалять линейные артефакты", value=True)
    use_median_filter = st.checkbox("Медианный фильтр 3×3", value=True)
    st.divider()
    st.subheader("📥 Экспорт результатов")
    export_mask = st.checkbox("Маска (PNG)", value=True)
    export_csv = st.checkbox("Таблица очагов (CSV)", value=True)
    st.divider()
    st.subheader("ℹ️ О модели")
    st.markdown("""
    - **Архитектура:** U-Net++
    - **Энкодер:** ResNet-34
    - **Dice (тест):** 0.74 ± 0.08
    - **Чувствительность:** 0.81
    - **Специфичность:** 0.99
    - **Обучающая выборка:** 130 снимков
    """)

uploaded_file = st.file_uploader(
    "Загрузите DICOM-файл сцинтиграфии (.IMA, .dcm)",
    type=['IMA', 'dcm']
)

if uploaded_file is not None:
    # Загружаем модель один раз, потом используем кеш
    model, device = load_model()

    # Предобработка
    st.info("Чтение DICOM, удаление контура врача...")
    dicom_bytes = uploaded_file.read()
    clean_img = remove_contour(dicom_bytes)
    h, w = clean_img.shape[:2]
    right = min(CROP_RIGHT, w - CROP_LEFT - 1)
    img_cropped = clean_img[CROP_TOP:h-CROP_BOTTOM, CROP_LEFT:w-right]

    # Инференс
    st.info("Запуск нейросети U-Net++... пожалуйста, подождите (≈15–20 сек)")
    prob_map = predict_sliding_window(img_cropped, model, device)

    # Постобработка
    pred_mask = (prob_map > threshold).astype(np.uint8) * 255
    if use_artifact_removal:
        pred_mask = remove_linear_artifacts(pred_mask)
    if use_median_filter:
        pred_mask = cv2.medianBlur(pred_mask, 3)

    # BSI
    bsi_zones, total_bsi = calculate_bsi(pred_mask, h, w)

    st.success("Анализ завершён!")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Результат сегментации")
        overlay = create_overlay(img_cropped, pred_mask)
        st.image(overlay, width=None, caption="Красный цвет — обнаруженные метастазы")
        with st.expander("Показать карту вероятностей"):
            fig, ax = plt.subplots()
            im = ax.imshow(prob_map, cmap='hot', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax)
            ax.axis('off')
            st.pyplot(fig)

    with col2:
        st.subheader("📋 Протокол BSI")
        bsi_data = [
            {"Зона": z['zone'], "BSI": f"{z['bsi']:.2f}%", "Пикселей": z['pixels']}
            for z in bsi_zones
        ]
        st.dataframe(bsi_data, hide_index=True, use_container_width=True)
        color = "red" if total_bsi > 1.0 else "green"
        st.markdown(
            f"<h2 style='text-align: center; color: {color};'>Итого: {total_bsi:.2f}%</h2>",
            unsafe_allow_html=True
        )
        if export_mask:
            _, mask_encoded = cv2.imencode('.png', pred_mask)
            st.download_button(
                "💾 Скачать маску (PNG)",
                mask_encoded.tobytes(),
                "pred_mask.png",
                "image/png"
            )
        if export_csv:
            csv_data = "zone,bsi,pixels\n" + "\n".join(
                f"{z['zone']},{z['bsi']},{z['pixels']}" for z in bsi_zones
            )
            st.download_button(
                "📊 Скачать таблицу (CSV)",
                csv_data,
                "bsi_report.csv",
                "text/csv"
            )
else:
    st.info("👆 Загрузите DICOM-файл, чтобы начать анализ")
