"""Componentes UI reutilizáveis para Streamlit."""

from __future__ import annotations

import cv2
import streamlit as st

from bodyfat import draw_measurement_overlay, BF_REFERENCE_TABLE


def render_bf_card(label: str, bf_data: dict):
    """Card compacto com BF%, categoria e medidas."""
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


def render_bodyfat_section(result, before_img, after_img, bf_before, bf_after, sex: str):
    """Seção completa de body fat com cards, delta, overlays e tabela."""
    col1, col2 = st.columns(2)

    if bf_before:
        with col1:
            render_bf_card("Antes", bf_before)
    if bf_after:
        with col2:
            render_bf_card("Depois", bf_after)

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
