"""Before/After Body Comparison — Streamlit UI."""

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison

from aligner import BodyAligner, draw_landmarks_overlay
from bodyfat import estimate_body_fat, draw_measurement_overlay, BF_REFERENCE_TABLE
from composer import OutputTemplate, compose  # noqa: F401
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
# Constants
# ---------------------------------------------------------------------------
VIEWS = [
    {"key": "front", "label": "Frente", "expanded": True},
    {"key": "side", "label": "Lateral", "expanded": False},
    {"key": "back", "label": "Costas", "expanded": False},
]
FILE_TYPES = ["jpg", "jpeg", "png", "heic", "webp"]

TEMPLATE_OPTIONS = {
    "Profissional": OutputTemplate.LABELED,
    "Limpo": OutputTemplate.CLEAN,
    "Instagram 1:1": OutputTemplate.SQUARE,
    "Instagram 4:5": OutputTemplate.PORTRAIT,
    "Story 9:16": OutputTemplate.STORY,
    "Com datas": OutputTemplate.WITH_DATES,
}


# ---------------------------------------------------------------------------
# Helper functions (defined BEFORE main flow)
# ---------------------------------------------------------------------------
def _preview(uploaded_file, caption: str):
    """Load image safely and show preview. Returns BGR numpy or None."""
    img = load_image(uploaded_file)
    if img is not None:
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)
    return img


def _render_bf_card(label, bf_data):
    cat = bf_data["category"]
    bf_pct = bf_data["ensemble_bf"]
    color_map = {"green": "🟢", "yellow": "🟡", "red": "🔴", "blue": "🔵"}
    icon = color_map.get(cat["color"], "⚪")

    st.metric(label=label, value=f"{bf_pct}%")
    st.write(f"{icon} {cat['label']}")

    if bf_data["navy_bf"]:
        st.caption(f"Navy: {bf_data['navy_bf']}% | Deurenberg: {bf_data['deurenberg_bf']}% | BMI: {bf_data['bmi']}")
    else:
        st.caption(f"Deurenberg: {bf_data['deurenberg_bf']}% | BMI: {bf_data['bmi']}")

    m = bf_data["measurements"]
    st.caption(f"Cintura: {m['waist_circ_cm']}cm | Quadril: {m['hip_circ_cm']}cm | Pescoco: {m['neck_circ_cm']}cm")


def _render_bodyfat_section(result, before_img, after_img, bf_before, bf_after, sex):
    """Render body fat details inside an expander."""
    col1, col2 = st.columns(2)

    if bf_before:
        with col1:
            _render_bf_card("Antes", bf_before)
    if bf_after:
        with col2:
            _render_bf_card("Depois", bf_after)

    # Delta
    if bf_before and bf_after:
        delta = bf_after["ensemble_bf"] - bf_before["ensemble_bf"]
        delta_str = f"{delta:+.1f}%"
        if delta < 0:
            st.success(f"Variacao: {delta_str} de gordura corporal")
        elif delta > 0:
            st.warning(f"Variacao: {delta_str} de gordura corporal")
        else:
            st.info("Sem variacao detectada")

    # Measurement overlay
    st.caption("Medicoes")
    bh, bw = before_img.shape[:2]
    ah, aw = after_img.shape[:2]
    m1, m2 = st.columns(2)
    if bf_before and result.landmarks_before is not None:
        with m1:
            ov = draw_measurement_overlay(before_img, result.landmarks_before, bw, bh)
            st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Antes", use_container_width=True)
    if bf_after and result.landmarks_after is not None:
        with m2:
            ov = draw_measurement_overlay(after_img, result.landmarks_after, aw, ah)
            st.image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB), caption="Depois", use_container_width=True)

    # Reference table
    with st.expander("Tabela de referencia"):
        ref = BF_REFERENCE_TABLE.get(sex, BF_REFERENCE_TABLE["M"])
        st.table({"Categoria": [r[0] for r in ref], "Faixa": [r[1] for r in ref]})

    st.caption(
        "Estimativa baseada em formulas antropometricas (Navy + Deurenberg). "
        "Precisao: ~4-6% vs DEXA. Para avaliacao clinica, consulte um profissional."
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Opcoes")

    template_name = st.radio("Estilo de saida", list(TEMPLATE_OPTIONS.keys()), index=0)
    selected_template = TEMPLATE_OPTIONS[template_name]

    # Labels (only for templates that show them)
    label_before = "Antes"
    label_after = "Depois"
    if selected_template not in (OutputTemplate.CLEAN,):
        label_before = st.text_input("Label antes", value="Antes")
        label_after = st.text_input("Label depois", value="Depois")

    st.divider()

    # Date overlay
    add_dates = st.toggle("Adicionar periodo", value=(selected_template == OutputTemplate.WITH_DATES))
    date_before = None
    date_after = None
    if add_dates:
        date_before = st.text_input("Data antes", placeholder="Jan 2025")
        date_after = st.text_input("Data depois", placeholder="Fev 2026")
        if selected_template != OutputTemplate.WITH_DATES:
            selected_template = OutputTemplate.WITH_DATES

    # Brightness matching
    color_match = st.toggle("Equalizar brilho", value=True,
                            help="Equaliza luminosidade entre as fotos")

    st.divider()

    # Advanced
    with st.expander("Avancado"):
        multi_view = st.toggle("3 angulos (Frente/Lateral/Costas)", value=False)
        bf_enabled = st.toggle("Estimar % gordura", value=False)
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
# Upload section
# ---------------------------------------------------------------------------
st.subheader("Envie suas fotos")

uploads = {}

if multi_view:
    for view in VIEWS:
        with st.expander(f"📷 {view['label']}", expanded=view["expanded"]):
            c1, c2 = st.columns(2)
            with c1:
                bf = st.file_uploader("Antes", type=FILE_TYPES, key=f"{view['key']}_before")
                bf_img = _preview(bf, "Antes") if bf else None
            with c2:
                af = st.file_uploader("Depois", type=FILE_TYPES, key=f"{view['key']}_after")
                af_img = _preview(af, "Depois") if af else None
            if bf_img is not None and af_img is not None:
                uploads[view["key"]] = {"before_img": bf_img, "after_img": af_img, "label": view["label"]}
else:
    c1, c2 = st.columns(2)
    with c1:
        before_file = st.file_uploader("Antes", type=FILE_TYPES, key="single_before")
        before_img_preview = _preview(before_file, "Antes") if before_file else None
    with c2:
        after_file = st.file_uploader("Depois", type=FILE_TYPES, key="single_after")
        after_img_preview = _preview(after_file, "Depois") if after_file else None
    if before_img_preview is not None and after_img_preview is not None:
        uploads["front"] = {"before_img": before_img_preview, "after_img": after_img_preview, "label": "Frente"}

# ---------------------------------------------------------------------------
# Auto-process
# ---------------------------------------------------------------------------
if not uploads:
    st.info("Envie pelo menos 1 par de fotos (antes + depois).")
    st.stop()

st.divider()

aligner = BodyAligner()

results = {}
for view_key, data in uploads.items():
    before_img = data["before_img"]
    after_img = data["after_img"]

    if before_img is None or after_img is None:
        st.error(f"Erro ao carregar imagem ({data['label']}). Tente outro formato.")
        continue

    # Color matching
    if color_match:
        after_img = match_brightness(after_img, before_img)

    with st.spinner(f"Alinhando {data['label']}..."):
        result = aligner.align(before_img, after_img)

    results[view_key] = {
        "result": result,
        "before_img": before_img,
        "after_img": after_img,
        "label": data["label"],
    }

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

        # Confidence (subtle — only show if low)
        if 0 < result.confidence < 0.4:
            st.caption(f"Confianca do alinhamento: {result.confidence:.0%} (baixa)")

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
            width=700,
        )

        # --- Compose & download ---
        st.subheader("Baixar")

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
        output_img.save(buf, format="PNG", quality=95)

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
                _render_bodyfat_section(
                    result, before_img, after_img,
                    bf_data_before, bf_data_after, bf_sex,
                )

        # --- Debug (expandable) ---
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
                st.info("Landmarks nao disponiveis (pose nao detectada).")
