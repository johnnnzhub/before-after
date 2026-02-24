"""Before/After Body Comparison — Streamlit UI."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from aligner import BodyAligner, draw_landmarks_overlay
from bodyfat import estimate_body_fat
from composer import OutputTemplate, compose
from ui_components import render_bodyfat_section
from utils import load_image, match_brightness

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Before & After", page_icon="📸", layout="centered")

_css_path = Path(__file__).parent / "style.css"
if _css_path.exists():
    st.markdown(f"<style>{_css_path.read_text()}</style>", unsafe_allow_html=True)

st.title("Before & After")
st.caption("by #cobaiateam")

# ---------------------------------------------------------------------------
# Debug mode (dev-only via ?debug=1)
# ---------------------------------------------------------------------------
debug_mode = st.query_params.get("debug", "0") == "1"

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def _get_aligner():
    """Load BodyAligner once — persists across reruns."""
    return BodyAligner()


def _file_cache_key(f) -> str:
    """Stable key from uploaded file metadata."""
    return f"{f.name}_{f.size}" if f else ""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VIEWS = [
    {"key": "front", "label": "Frente", "expanded": True},
    {"key": "side", "label": "Lateral", "expanded": False},
    {"key": "back", "label": "Costas", "expanded": False},
]
FILE_TYPES = ["jpg", "jpeg", "png", "heic", "webp"]
MAX_CACHED_ALIGNMENTS = 10


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _preview(uploaded_file, caption: str):
    """Load image safely and show preview. Returns BGR numpy or None."""
    try:
        img = load_image(uploaded_file)
    except ValueError:
        st.error("Formato HEIC não suportado neste ambiente. Converta para JPG ou PNG.")
        return None
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
    return img


# ---------------------------------------------------------------------------
# Upload section (always visible)
# ---------------------------------------------------------------------------
st.subheader("Envie suas fotos")

multi_view = st.toggle("3 ângulos (Frente/Lateral/Costas)", value=False,
                       help="Compare frente, lateral e costas")

uploads = {}

if multi_view:
    for view in VIEWS:
        with st.expander(f"📷 {view['label']}", expanded=view["expanded"]):
            c1, c2 = st.columns(2)
            with c1:
                bf = st.file_uploader("Antes", type=FILE_TYPES, key=f"{view['key']}_before",
                                      help="JPG, PNG ou HEIC. Máximo 20MB.")
                bf_img = _preview(bf, "Antes") if bf else None
            with c2:
                af = st.file_uploader("Depois", type=FILE_TYPES, key=f"{view['key']}_after",
                                      help="JPG, PNG ou HEIC. Máximo 20MB.")
                af_img = _preview(af, "Depois") if af else None
            if bf_img is not None and af_img is not None:
                uploads[view["key"]] = {
                    "before_img": bf_img, "after_img": af_img,
                    "label": view["label"],
                    "cache_key": f"{view['key']}_{_file_cache_key(bf)}_{_file_cache_key(af)}",
                }
else:
    c1, c2 = st.columns(2)
    with c1:
        before_file = st.file_uploader("Antes", type=FILE_TYPES, key="single_before",
                                       help="JPG, PNG ou HEIC. Máximo 20MB.")
        before_img_preview = _preview(before_file, "Antes") if before_file else None
    with c2:
        after_file = st.file_uploader("Depois", type=FILE_TYPES, key="single_after",
                                      help="JPG, PNG ou HEIC. Máximo 20MB.")
        after_img_preview = _preview(after_file, "Depois") if after_file else None
    if before_img_preview is not None and after_img_preview is not None:
        uploads["front"] = {
            "before_img": before_img_preview, "after_img": after_img_preview,
            "label": "Frente",
            "cache_key": f"front_{_file_cache_key(before_file)}_{_file_cache_key(after_file)}",
        }

# ---------------------------------------------------------------------------
# Stop here if no uploads — sidebar won't render yet
# ---------------------------------------------------------------------------
if not uploads:
    st.info("Envie um par de fotos para começar.")
    st.stop()

# ---------------------------------------------------------------------------
# Sidebar (appears only after upload)
# ---------------------------------------------------------------------------
selected_template = OutputTemplate.LABELED

with st.sidebar:
    st.header("Opções")

    label_before = st.text_input("Label antes", value="Antes")
    label_after = st.text_input("Label depois", value="Depois")

    st.divider()

    add_dates = st.toggle("Adicionar período", value=False)
    date_before = None
    date_after = None
    if add_dates:
        date_before = st.text_input("Data antes", placeholder="Jan 2025")
        date_after = st.text_input("Data depois", placeholder="Fev 2026")
        selected_template = OutputTemplate.WITH_DATES

    color_match = st.toggle("Equalizar brilho", value=True,
                            help="Equaliza luminosidade entre as fotos")

    st.divider()

    bf_enabled = st.toggle("Estimar % gordura corporal", value=False,
                           help="Estimativa Navy + Deurenberg (~4-6% precisão)")
    bf_sex = "M"
    bf_age = 30
    bf_height = 175.0
    bf_weight = 75.0
    if bf_enabled:
        bf_sex = st.selectbox("Sexo", ["M", "F"])
        bf_age = st.number_input("Idade", min_value=10, max_value=100, value=30)
        bf_height = st.number_input("Altura (cm)", min_value=100.0, max_value=250.0, value=175.0, step=0.5)
        bf_weight = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=75.0, step=0.5)

# ---------------------------------------------------------------------------
# Auto-process (with session_state cache)
# ---------------------------------------------------------------------------
st.divider()

aligner = _get_aligner()

results = {}
for view_key, data in uploads.items():
    before_img = data["before_img"]
    after_img = data["after_img"]

    if before_img is None or after_img is None:
        st.error(f"Erro ao carregar imagem ({data['label']}). Tente outro formato.")
        continue

    # Cache key includes color_match because brightness affects alignment input
    full_key = f"align_{data['cache_key']}_{color_match}"

    if full_key in st.session_state:
        results[view_key] = st.session_state[full_key]
    else:
        if color_match:
            after_img = match_brightness(after_img, before_img)

        with st.spinner(f"Alinhando {data['label']}..."):
            result = aligner.align(before_img, after_img)

        entry = {
            "result": result,
            "before_img": before_img,
            "after_img": after_img,
            "label": data["label"],
        }
        st.session_state[full_key] = entry
        results[view_key] = entry

    # Session state LRU cleanup
    align_keys = [k for k in st.session_state if k.startswith("align_")]
    if len(align_keys) > MAX_CACHED_ALIGNMENTS:
        for old_key in align_keys[:-MAX_CACHED_ALIGNMENTS]:
            del st.session_state[old_key]

if not results:
    st.stop()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
view_keys = list(results.keys())
view_labels = [results[k]["label"] for k in view_keys]

if len(view_keys) == 1:
    containers = {view_keys[0]: st.container()}
else:
    outer_tabs = st.tabs(view_labels)
    containers = {k: t for k, t in zip(view_keys, outer_tabs)}

for view_key, container in containers.items():
    r = results[view_key]
    result = r["result"]
    before_img = r["before_img"]
    after_img = r["after_img"]

    with container:
        # Warnings
        for w in result.warnings:
            st.warning(w)

        # Confidence indicator (unified)
        if result.confidence > 0:
            if result.confidence >= 0.7:
                st.caption(f"Alinhamento: {result.confidence:.0%}")
            elif result.confidence >= 0.4:
                st.caption(f"Alinhamento: {result.confidence:.0%} (moderado)")
            else:
                st.caption(f"Alinhamento: {result.confidence:.0%} (baixo)")

        # --- Slider comparison ---
        before_rgb = cv2.cvtColor(result.before_aligned, cv2.COLOR_BGR2RGB)
        after_rgb = cv2.cvtColor(result.after_aligned, cv2.COLOR_BGR2RGB)

        image_comparison(
            img1=Image.fromarray(before_rgb),
            img2=Image.fromarray(after_rgb),
            label1=label_before,
            label2=label_after,
            starting_position=50,
            show_labels=True,
        )

        # --- Compose & download ---
        # Body fat data (if enabled) — only "depois" goes on the photo
        bf_data_before = None
        bf_data_after = None
        if bf_enabled:
            bh, bw = before_img.shape[:2]
            ah, aw = after_img.shape[:2]
            if result.landmarks_before is not None:
                bf_data_before = estimate_body_fat(
                    result.landmarks_before, bw, bh, bf_height, bf_weight, bf_age, bf_sex
                )
            if result.landmarks_after is not None:
                bf_data_after = estimate_body_fat(
                    result.landmarks_after, aw, ah, bf_height, bf_weight, bf_age, bf_sex
                )

        output_img = compose(
            before=result.before_aligned,
            after=result.after_aligned,
            template=selected_template,
            label_before=label_before,
            label_after=label_after,
            date_before=date_before if add_dates else None,
            date_after=date_after if add_dates else None,
            bf_after=bf_data_after if bf_enabled else None,
        )

        st.image(output_img, use_container_width=True)

        buf = io.BytesIO()
        output_img.save(buf, format="PNG")

        st.download_button(
            label="Baixar foto",
            data=buf.getvalue(),
            file_name=f"before_after_{view_key}.png",
            mime="image/png",
            key=f"dl_{view_key}",
            type="primary",
            use_container_width=True,
        )

        # --- Body fat details (expandable) ---
        if bf_enabled and (bf_data_before or bf_data_after):
            with st.expander("% Gordura corporal"):
                render_bodyfat_section(
                    result, before_img, after_img,
                    bf_data_before, bf_data_after, bf_sex,
                )

        # --- Debug (dev-only via ?debug=1) ---
        if debug_mode:
            with st.expander("Debug (landmarks)"):
                if result.landmarks_before is not None and result.landmarks_after is not None:
                    d1, d2 = st.columns(2)
                    with d1:
                        ov = draw_landmarks_overlay(before_img, result.landmarks_before)
                        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Antes", use_container_width=True)
                    with d2:
                        ov = draw_landmarks_overlay(after_img, result.landmarks_after)
                        st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Depois", use_container_width=True)
                else:
                    st.info("Landmarks não disponíveis (pose não detectada).")
