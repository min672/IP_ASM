"""
app.py — Fingerprint Enhancement System: Data Analysis Dashboard (Streamlit)

Implements the assignment's shared requirements:
  - Data Analysis Dashboard: summarises object properties and
    inspection/enhancement results
  - (Extra Efforts) GUI: bulk image ingestion (multi-file upload)
  - (Extra Efforts) Reporting: automated PDF export

Run in Colab with:
    !streamlit run app.py &>/content/logs.txt &
    !npx localtunnel --port 8501
(see the README cell for the full Colab launch instructions)
"""

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from fpdf import FPDF

from pipeline import run_pipeline, TARGET_SIZE

st.set_page_config(page_title="Fingerprint Enhancement System", layout="wide")


# =============================================================================
# Helpers
# =============================================================================

def load_image_from_upload(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    return image


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return buf


def show_gray(image, caption=""):
    st.image(image, caption=caption, clamp=True, channels="GRAY" if image.ndim == 2 else "RGB",
              use_container_width=True)


def minutiae_overlay_figure(skeleton, true_minutiae, false_minutiae, title=""):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.imshow(skeleton, cmap="gray")
    for m in false_minutiae:
        ax.plot(m["x"], m["y"], "x", color="yellow", markersize=5, markeredgewidth=1)
    for m in true_minutiae:
        c = "red" if m["type"] == "ending" else "blue"
        ax.plot(m["x"], m["y"], "o", color=c, markersize=5)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    return fig


def build_pdf_report(single_results=None, batch_df=None, batch_chart_png=None):
    """Build a simple PDF report (Extra Efforts: Reporting)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Fingerprint Enhancement System - Report", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, "Module 4 - Feature Analysis Dashboard Export", ln=True)
    pdf.ln(4)

    if single_results is not None:
        d = single_results["member_d"]
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Single-Image Analysis", ln=True)
        pdf.set_font("Helvetica", "", 10)
        rows = [
            ("SSIM (raw vs enhanced)", d["SSIM_raw_vs_enhanced"]),
            ("PSNR (raw vs enhanced)", f"{d['PSNR_raw_vs_enhanced_dB']} dB"),
            ("Raw minutiae", d["raw_minutiae"]),
            ("True minutiae", d["true_minutiae"]),
            ("False minutiae removed", d["false_minutiae_removed"]),
            ("Ridge endings", d["ridge_endings"]),
            ("Bifurcations", d["bifurcations"]),
            ("Ridge density", d["ridge_density"]),
            ("Quality score (0-100)", d["quality_score"]),
            ("Suitable for matching", "Yes" if d["suitable_for_matching"] else "No"),
            ("mm per pixel", d["mm_per_px"]),
            ("Avg ridge spacing", f"{d['avg_ridge_spacing_mm']} mm"),
            ("Calibrated image size", f"{d['image_size_mm']} mm"),
        ]
        for label, value in rows:
            pdf.cell(90, 7, str(label), border=1)
            pdf.cell(0, 7, str(value), border=1, ln=True)
        pdf.ln(6)

    if batch_df is not None:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Batch Analysis Summary", ln=True)
        pdf.set_font("Helvetica", "", 9)
        summary = batch_df.groupby("alteration_level").mean(numeric_only=True).round(3)
        col_names = ["level"] + list(summary.columns)
        col_width = 190 / len(col_names)
        pdf.set_font("Helvetica", "B", 8)
        for name in col_names:
            pdf.cell(col_width, 7, str(name)[:14], border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 8)
        for level, row in summary.iterrows():
            pdf.cell(col_width, 7, str(level), border=1)
            for value in row:
                pdf.cell(col_width, 7, str(value), border=1)
            pdf.ln()
        pdf.ln(6)

        if batch_chart_png is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(batch_chart_png.getvalue())
                tmp_path = tmp.name
            pdf.image(tmp_path, w=180)

    return bytes(pdf.output(dest="S"))


# =============================================================================
# Sidebar — mode selection + bulk ingestion
# =============================================================================

st.sidebar.title("Fingerprint Enhancement System")
mode = st.sidebar.radio("Mode", ["Single image", "Batch analysis (bulk upload)"])

st.sidebar.markdown("---")
st.sidebar.caption(
    "Pipeline: Preprocessing (A) -> Ridge Recovery (B) -> "
    "Ridge Structure Extraction (C) -> Feature Analysis (D)"
)


# =============================================================================
# MODE 1 — Single image, full stage-by-stage dashboard
# =============================================================================

if mode == "Single image":
    st.title("Single Image Dashboard")
    uploaded = st.file_uploader("Upload a fingerprint image", type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded is not None:
        raw_image = load_image_from_upload(uploaded)
        if raw_image is None:
            st.error("Could not read this file as an image.")
        else:
            with st.spinner("Running full pipeline (Preprocessing -> Enhancement -> Skeleton -> Feature Analysis)..."):
                try:
                    result = run_pipeline(raw_image)
                except Exception as e:
                    st.error(f"Pipeline failed on this image: {e}")
                    result = None

            if result is not None:
                d = result["member_d"]

                st.subheader("Pipeline Stages")
                cols = st.columns(4)
                with cols[0]:
                    show_gray(result["grayscale"], "1. Grayscale (Preprocessing)")
                with cols[1]:
                    show_gray(result["clahe"], "2. CLAHE Enhanced (Member A)")
                with cols[2]:
                    show_gray(result["member_b_output"], "3. Gabor Ridge Recovery (Member B)")
                with cols[3]:
                    show_gray(result["skeleton_255"], "4. Skeleton (Member C)")

                st.subheader("Minutiae Detection (Member D)")
                fig = minutiae_overlay_figure(
                    result["skeleton_255"], d["true_minutiae_list"], d["false_minutiae_list"],
                    title=f"{d['ridge_endings']} ending, {d['bifurcations']} bifurcation "
                          f"(red/blue=true, yellow x=false)"
                )
                minutiae_png = fig_to_png_bytes(fig)
                st.pyplot(fig, use_container_width=False)

                st.subheader("Metrics Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("SSIM (raw vs enhanced)", d["SSIM_raw_vs_enhanced"])
                m2.metric("PSNR (dB)", d["PSNR_raw_vs_enhanced_dB"])
                m3.metric("True minutiae", d["true_minutiae"])
                m4.metric("Quality score", d["quality_score"])

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Ridge endings", d["ridge_endings"])
                m6.metric("Bifurcations", d["bifurcations"])
                m7.metric("Suitable for matching", "Yes" if d["suitable_for_matching"] else "No")
                m8.metric("Ridge spacing (mm)", d["avg_ridge_spacing_mm"])

                with st.expander("Full details / Image Calibration"):
                    st.json({k: v for k, v in d.items()
                             if k not in ("true_minutiae_list", "false_minutiae_list")})

                st.subheader("Export")
                pdf_bytes = build_pdf_report(single_results=result)
                st.download_button(
                    "Download PDF Report", data=pdf_bytes,
                    file_name="fingerprint_report.pdf", mime="application/pdf",
                )


# =============================================================================
# MODE 2 — Batch analysis (bulk ingestion + aggregated statistics)
# =============================================================================

else:
    st.title("Batch Analysis")
    st.caption("Upload multiple images at once. Name files so the alteration level "
               "can be detected (e.g. containing 'Easy', 'Medium', 'Hard', or default 'Real').")

    uploaded_files = st.file_uploader(
        "Upload multiple fingerprint images",
        type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    def detect_level(filename):
        name = filename.lower()
        for level in ["easy", "medium", "hard"]:
            if level in name:
                return level.capitalize()
        return "Real"

    if uploaded_files:
        st.write(f"{len(uploaded_files)} files uploaded.")

        if st.button("Run batch analysis"):
            rows = []
            progress = st.progress(0)
            status = st.empty()
            failures = []

            for i, uploaded in enumerate(uploaded_files):
                status.text(f"Processing {uploaded.name} ({i+1}/{len(uploaded_files)})...")
                raw_image = load_image_from_upload(uploaded)
                if raw_image is not None:
                    try:
                        result = run_pipeline(raw_image)
                        d = result["member_d"]
                        rows.append({
                            "filename": uploaded.name,
                            "alteration_level": detect_level(uploaded.name),
                            "SSIM": d["SSIM_raw_vs_enhanced"],
                            "PSNR_dB": d["PSNR_raw_vs_enhanced_dB"],
                            "true_minutiae": d["true_minutiae"],
                            "ridge_density": d["ridge_density"],
                            "quality_score": d["quality_score"],
                            "suitable_for_matching": d["suitable_for_matching"],
                        })
                    except Exception as e:
                        failures.append((uploaded.name, str(e)))
                progress.progress((i + 1) / len(uploaded_files))

            status.empty()
            progress.empty()

            if failures:
                with st.expander(f"{len(failures)} image(s) failed to process"):
                    for name, err in failures:
                        st.write(f"- {name}: {err}")

            if rows:
                df = pd.DataFrame(rows)
                st.session_state["batch_df"] = df
                st.success(f"Processed {len(df)} images successfully.")

        if "batch_df" in st.session_state:
            df = st.session_state["batch_df"]

            st.subheader("Per-image Results")
            st.dataframe(df, use_container_width=True)

            st.subheader("Aggregated by Alteration Level")
            summary = df.groupby("alteration_level").agg(
                n=("filename", "count"),
                mean_SSIM=("SSIM", "mean"),
                mean_PSNR_dB=("PSNR_dB", "mean"),
                mean_true_minutiae=("true_minutiae", "mean"),
                mean_quality_score=("quality_score", "mean"),
            ).round(3)
            level_order = [lv for lv in ["Real", "Easy", "Medium", "Hard"] if lv in summary.index]
            summary = summary.reindex(level_order)
            st.dataframe(summary, use_container_width=True)

            st.subheader("Trend Charts")
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            summary["mean_SSIM"].plot(kind="bar", ax=axes[0], color="steelblue")
            axes[0].set_title("Mean SSIM by Level")
            summary["mean_true_minutiae"].plot(kind="bar", ax=axes[1], color="indianred")
            axes[1].set_title("Mean True Minutiae by Level")
            summary["mean_quality_score"].plot(kind="bar", ax=axes[2], color="seagreen")
            axes[2].set_title("Mean Quality Score by Level")
            plt.tight_layout()
            st.pyplot(fig)
            chart_png = fig_to_png_bytes(fig)

            st.subheader("Export")
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")

            pdf_bytes = build_pdf_report(batch_df=df, batch_chart_png=chart_png)
            st.download_button(
                "Download PDF Report", data=pdf_bytes,
                file_name="batch_report.pdf", mime="application/pdf",
            )
