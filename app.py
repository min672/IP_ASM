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


def show_minutiae_legend():
    st.markdown(
        "🔵 **Bifurcation (True)**&nbsp;&nbsp;&nbsp;"
        "🔴 **Ridge Ending (True)**&nbsp;&nbsp;&nbsp;"
        "🟡 **False (removed)**"
    )


def render_threshold_sidebar():
    """
    Sliders for every tunable stage, organised by member, so parameters
    can be adjusted live instead of being hardcoded. Returns the four
    kwargs dicts run_pipeline() expects.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pipeline Parameters")

    with st.sidebar.expander("Member A - Preprocessing / Enhancement"):
        denoising_method = st.radio("Denoising method", ["median", "gaussian"], index=0, key="a_denoise")
        clahe_clip_limit = st.slider("CLAHE clip limit", 1.0, 6.0, 2.0, 0.1, key="a_clip")
        clahe_tile = st.slider("CLAHE tile grid size", 2, 16, 8, 1, key="a_tile")
        target_mean = st.slider("Target mean brightness", 80.0, 180.0, 128.0, 1.0, key="a_mean")
        target_std = st.slider("Target contrast (std)", 20.0, 80.0, 50.0, 1.0, key="a_std")

    with st.sidebar.expander("Member B - ROI / Orientation / Gabor"):
        roi_window_size = st.slider("ROI window size", 5, 31, 15, 2, key="b_roiwin")
        roi_texture_percentile = st.slider("ROI texture percentile", 0.0, 100.0, 50.0, 5.0, key="b_roipct")
        orientation_sigma = st.slider("Orientation smoothing (sigma)", 1.0, 8.0, 4.0, 0.5, key="b_sigma")
        minimum_coherence = st.slider("Minimum orientation coherence", 0.0, 0.6, 0.20, 0.01, key="b_coh")
        gabor_kernel_size = st.slider("Gabor kernel size", 9, 41, 25, 2, key="b_gk")

    with st.sidebar.expander("Member C - Threshold / Morphology"):
        adaptive_block_size = st.slider("Adaptive threshold block size", 5, 31, 13, 2, key="c_block")
        adaptive_C = st.slider("Adaptive threshold C", 0, 15, 5, 1, key="c_C")
        morph_kernel_size = st.slider("Morphology kernel size", 1, 7, 3, 2, key="c_morph")

    with st.sidebar.expander("Member D - Minutiae Filtering"):
        min_ridge_length = st.slider("Min ridge length (spur rejection)", 1, 20, 6, 1, key="d_minlen")
        min_pair_distance = st.slider("Min pair distance (bridge rejection)", 1, 30, 10, 1, key="d_pairdist")
        boundary_margin = st.slider("Boundary margin", 1, 30, 10, 1, key="d_boundary")
        max_neighbours = st.slider("Max neighbours (density filter)", 1, 15, 5, 1, key="d_maxneigh")
        neighbour_radius = st.slider("Neighbour radius (density filter)", 5, 40, 15, 1, key="d_neighrad")
        max_straightness = st.slider("Max straightness (1.0 = perfectly straight)", 0.80, 1.00, 0.97, 0.01, key="d_straight")
        min_usable_minutiae = st.slider("Min usable minutiae (matching threshold)", 1, 40, 12, 1, key="d_minusable")

    member_a_kwargs = dict(
        denoising_method=denoising_method,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_grid=(clahe_tile, clahe_tile),
        target_mean=target_mean,
        target_std=target_std,
    )
    member_b_kwargs = dict(
        roi_window_size=roi_window_size,
        roi_texture_percentile=roi_texture_percentile,
        orientation_sigma=orientation_sigma,
        minimum_coherence=minimum_coherence,
        gabor_kernel_size=gabor_kernel_size,
    )
    member_c_kwargs = dict(
        adaptive_block_size=adaptive_block_size,
        adaptive_C=adaptive_C,
        morph_kernel_size=morph_kernel_size,
    )
    member_d_kwargs = dict(
        min_ridge_length=min_ridge_length,
        min_pair_distance=min_pair_distance,
        boundary_margin=boundary_margin,
        max_neighbours=max_neighbours,
        neighbour_radius=neighbour_radius,
        max_straightness=max_straightness,
        min_usable_minutiae=min_usable_minutiae,
    )
    return member_a_kwargs, member_b_kwargs, member_c_kwargs, member_d_kwargs


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
            ("SSIM (raw vs Member B output)", d["SSIM_raw_vs_enhanced"]),
            ("PSNR (raw vs enhanced)", f"{d['PSNR_raw_vs_enhanced_dB']} dB"),
            ("OCL before -> after", f"{d.get('OCL_before', 'N/A')} -> {d.get('OCL_after', 'N/A')}"),
            ("LCS before -> after", f"{d.get('LCS_before', 'N/A')} -> {d.get('LCS_after', 'N/A')}"),
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
st.sidebar.caption(
    "Pipeline: Preprocessing (A) -> Ridge Recovery (B) -> "
    "Ridge Structure Extraction (C) -> Feature Analysis (D)"
)

a_kwargs, b_kwargs, c_kwargs, d_kwargs = render_threshold_sidebar()


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
                    result = run_pipeline(
                        raw_image,
                        member_a_kwargs=a_kwargs,
                        member_b_kwargs=b_kwargs,
                        member_c_kwargs=c_kwargs,
                        member_d_kwargs=d_kwargs,
                    )
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
                show_minutiae_legend()
                fig = minutiae_overlay_figure(
                    result["skeleton_255"], d["true_minutiae_list"], d["false_minutiae_list"],
                    title=f"{d['ridge_endings']} ending, {d['bifurcations']} bifurcation"
                )
                minutiae_png = fig_to_png_bytes(fig)
                st.pyplot(fig, use_container_width=False)

                st.subheader("Metrics Summary")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("SSIM (raw vs Member B output)", d["SSIM_raw_vs_enhanced"])
                m2.metric("PSNR (raw vs Member B, dB)", d["PSNR_raw_vs_enhanced_dB"])
                m3.metric("True minutiae", d["true_minutiae"])
                m4.metric("Quality score", d["quality_score"])

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Ridge endings", d["ridge_endings"])
                m6.metric("Bifurcations", d["bifurcations"])
                m7.metric("Suitable for matching", "Yes" if d["suitable_for_matching"] else "No")
                m8.metric("Ridge spacing (mm)", d["avg_ridge_spacing_mm"])

                m9, m10, m11, m12 = st.columns(4)
                m9.metric("OCL before", d.get("OCL_before", "N/A"))
                m10.metric("OCL after", d.get("OCL_after", "N/A"),
                           delta=d.get("OCL_improvement"))
                m11.metric("LCS before", d.get("LCS_before", "N/A"))
                m12.metric("LCS after", d.get("LCS_after", "N/A"),
                           delta=d.get("LCS_improvement"))

                st.subheader("Before vs After — What Changed, and Where")

                bar_fig, bar_ax = plt.subplots(figsize=(5, 3.5))
                labels = ["OCL\n(orientation certainty)", "LCS\n(local clarity)"]
                before_vals = [d.get("OCL_before", 0), d.get("LCS_before", 0)]
                after_vals = [d.get("OCL_after", 0), d.get("LCS_after", 0)]
                x = np.arange(len(labels))
                width = 0.35
                bar_ax.bar(x - width / 2, before_vals, width, label="Before (raw)", color="lightgray")
                bar_ax.bar(x + width / 2, after_vals, width, label="After (Member B output)", color="steelblue")
                bar_ax.set_ylim(0, 1)
                bar_ax.set_xticks(x)
                bar_ax.set_xticklabels(labels)
                bar_ax.set_ylabel("Score (0-1, higher = better)")
                bar_ax.legend(fontsize=8)
                bar_ax.set_title("Quality metrics: before vs after", fontsize=10)
                st.pyplot(bar_fig)
                st.caption(
                    "A taller blue bar than gray bar means the enhancement pipeline "
                    "(Member A + B) genuinely improved that aspect of the fingerprint - "
                    "not just changed how it looks."
                )

                st.markdown("**Where did orientation certainty (OCL) change?**")
                ocl_cols = st.columns(2)
                with ocl_cols[0]:
                    fig_o1, ax_o1 = plt.subplots(figsize=(4, 4))
                    im1 = ax_o1.imshow(d["OCL_before_map"], cmap="viridis", vmin=0, vmax=1)
                    ax_o1.set_title("Before (raw)", fontsize=9)
                    ax_o1.axis("off")
                    st.pyplot(fig_o1)
                with ocl_cols[1]:
                    fig_o2, ax_o2 = plt.subplots(figsize=(4, 4))
                    im2 = ax_o2.imshow(d["OCL_after_map"], cmap="viridis", vmin=0, vmax=1)
                    ax_o2.set_title("After (Member B output)", fontsize=9)
                    ax_o2.axis("off")
                    st.pyplot(fig_o2)
                st.caption(
                    "Brighter (yellow) = orientation direction estimated with high confidence. "
                    "Darker (purple) = unreliable region. Look for areas that turned brighter "
                    "after processing - that's where Gabor recovery clarified the ridge direction."
                )

                st.markdown("**Where did local clarity (LCS) change?**")
                lcs_cols = st.columns(2)
                with lcs_cols[0]:
                    fig_l1, ax_l1 = plt.subplots(figsize=(4, 4))
                    ax_l1.imshow(d["LCS_before_map"], cmap="viridis", vmin=0, vmax=1)
                    ax_l1.set_title("Before (raw)", fontsize=9)
                    ax_l1.axis("off")
                    st.pyplot(fig_l1)
                with lcs_cols[1]:
                    fig_l2, ax_l2 = plt.subplots(figsize=(4, 4))
                    ax_l2.imshow(d["LCS_after_map"], cmap="viridis", vmin=0, vmax=1)
                    ax_l2.set_title("After (Member B output)", fontsize=9)
                    ax_l2.axis("off")
                    st.pyplot(fig_l2)
                st.caption(
                    "Brighter (yellow) = ridge and valley pixels are clearly separated in that "
                    "block. Darker (purple) = blurred, ridges and valleys blend together."
                )

                with st.expander("Full details / Image Calibration"):
                    st.json({k: v for k, v in d.items()
                             if k not in ("true_minutiae_list", "false_minutiae_list",
                                          "OCL_before_map", "OCL_after_map",
                                          "LCS_before_map", "LCS_after_map")})

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
    st.caption("Upload images into the correct category box below. Each box "
               "supports multiple files at once.")

    CATEGORIES = ["Real", "Easy", "Medium", "Hard"]  # edit this list if your
                                                       # dataset has 3 categories
                                                       # instead of 4, e.g.
                                                       # ["Real", "Altered-Easy", "Altered-Hard"]

    category_files = {}
    tabs = st.tabs(CATEGORIES)
    for tab, category in zip(tabs, CATEGORIES):
        with tab:
            category_files[category] = st.file_uploader(
                f"Upload {category} images",
                type=["bmp", "png", "jpg", "jpeg", "tif", "tiff"],
                accept_multiple_files=True,
                key=f"uploader_{category}",
            )

    total_files = sum(len(files or []) for files in category_files.values())
    st.write(f"{total_files} files uploaded across {len(CATEGORIES)} categories.")
    for category in CATEGORIES:
        count = len(category_files[category] or [])
        st.caption(f"- {category}: {count} file(s)")

    if total_files > 0:
        if st.button("Run batch analysis"):
            rows = []
            progress = st.progress(0)
            status = st.empty()
            failures = []
            processed = 0

            for category, files in category_files.items():
                for uploaded in (files or []):
                    processed += 1
                    status.text(f"Processing {uploaded.name} [{category}] ({processed}/{total_files})...")
                    raw_image = load_image_from_upload(uploaded)
                    if raw_image is not None:
                        try:
                            result = run_pipeline(
                                raw_image,
                                member_a_kwargs=a_kwargs,
                                member_b_kwargs=b_kwargs,
                                member_c_kwargs=c_kwargs,
                                member_d_kwargs=d_kwargs,
                            )
                            d = result["member_d"]
                            rows.append({
                                "filename": uploaded.name,
                                "alteration_level": category,
                                "SSIM": d["SSIM_raw_vs_enhanced"],
                                "PSNR_dB": d["PSNR_raw_vs_enhanced_dB"],
                                "OCL_before": d.get("OCL_before", None),
                                "OCL_after": d.get("OCL_after", None),
                                "LCS_before": d.get("LCS_before", None),
                                "LCS_after": d.get("LCS_after", None),
                                "true_minutiae": d["true_minutiae"],
                                "ridge_density": d["ridge_density"],
                                "quality_score": d["quality_score"],
                                "suitable_for_matching": d["suitable_for_matching"],
                            })
                        except Exception as e:
                            failures.append((uploaded.name, category, str(e)))
                    progress.progress(processed / total_files)

            status.empty()
            progress.empty()

            if failures:
                with st.expander(f"{len(failures)} image(s) failed to process"):
                    for name, category, err in failures:
                        st.write(f"- [{category}] {name}: {err}")

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
