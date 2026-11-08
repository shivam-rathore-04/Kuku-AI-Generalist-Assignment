from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import zipfile

from dotenv import load_dotenv
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from candidate_generator import CandidateGenerator
from evaluator import evaluate_candidate
from parser import ScriptParser
from ranker import rank_promos


load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def _safe_filename(name: str) -> str:
    stem = Path(name).stem if name else "promos"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "promos"


def _render_score_breakdown(candidate) -> None:
    ev = candidate.evaluation
    if ev is None:
        return

    hook_contrib = 0.45 * ev.hook_score
    cliff_contrib = 0.35 * ev.cliffhanger_score
    clarity_contrib = 0.20 * ev.clarity_score

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Hook**")
        st.progress(ev.hook_score / 10.0)
        st.caption(f"Score: {ev.hook_score}/10")
    with c2:
        st.markdown("**Cliffhanger**")
        st.progress(ev.cliffhanger_score / 10.0)
        st.caption(f"Score: {ev.cliffhanger_score}/10")
    with c3:
        st.markdown("**Clarity**")
        st.progress(ev.clarity_score / 10.0)
        st.caption(f"Score: {ev.clarity_score}/10")


def _candidate_payload(parsed, source_file: str, ranked_promos: list) -> dict:
    return {
        "episode_title": parsed.title,
        "logline": parsed.logline,
        "source_file": source_file,
        "total_parsed_scenes": parsed.scene_count,
        "duration_sec": round(parsed.total_duration_sec, 2),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "promos": [promo.to_dict() for promo in ranked_promos],
    }


def _build_pdf_report(export_payload: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
        title="Kuku AI Promo Report",
        author="Kuku AI Generalist Assignment",
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    section_style = styles["Heading2"]
    sub_style = styles["Heading3"]
    body = styles["BodyText"]
    body.spaceAfter = 6

    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=9,
        leading=11,
        spaceAfter=4,
    )

    story = []
    story.append(Paragraph("Kuku AI Promo Report", title_style))
    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Batch summary for top ranked microdrama promos.", body))
    story.append(Spacer(1, 0.12 * inch))

    summary_data = [
        ["Generated", export_payload.get("generated_at_utc", "")],
        ["Episodes", str(export_payload.get("episodes", 0))],
        ["Batch sources", str(len(export_payload.get("batch_source", [])))],
        ["Promos in report", str(len(export_payload.get("promos", [])))],
    ]
    summary_table = Table(summary_data, colWidths=[1.5 * inch, 5.8 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("How to read this report", section_style))
    story.append(Paragraph(
        "Each promo is a candidate clip extracted from the batch, scored for hook, cliffhanger, and clarity. "
        "The episode label shows which source file produced it.",
        body,
    ))
    story.append(Spacer(1, 0.1 * inch))

    for idx, promo in enumerate(export_payload.get("promos", []), start=1):
        story.append(Paragraph(f"Promo #{idx}: {escape(str(promo.get('episode_title', 'Untitled')))}", section_style))
        story.append(Spacer(1, 0.06 * inch))

        promo_summary = [
            ["Source file", promo.get("source_file", "")],
            ["Episode", promo.get("episode_title", "")],
            ["Composite score", f"{promo.get('composite_score', 0):.2f}"],
            ["Duration", f"{promo.get('duration_sec', 0):.1f} sec"],
            ["Start → End", f"{promo.get('start_timestamp_fmt', '')} → {promo.get('end_timestamp_fmt', '')}"],
        ]
        promo_table = Table(promo_summary, colWidths=[1.4 * inch, 6.0 * inch])
        promo_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("LEADING", (0, 0), (-1, -1), 11),
                ]
            )
        )
        story.append(promo_table)
        story.append(Spacer(1, 0.06 * inch))

        story.append(Paragraph(f"Hook: {escape(str(promo.get('evaluation', {}).get('hook_element', '')))}", body))
        story.append(Paragraph(f"Cliffhanger: {escape(str(promo.get('evaluation', {}).get('cliffhanger_element', '')))}", body))
        story.append(Paragraph(f"Justification: {escape(str(promo.get('evaluation', {}).get('justification', '')))}", body))

        flags = promo.get("diagnostics") or []
        story.append(Paragraph("Quality flags: " + (", ".join(map(str, flags)) if flags else "None"), small))
        story.append(Spacer(1, 0.12 * inch))

    def _page_number(canvas, doc_obj):
        canvas.setFont("Helvetica", 9)
        canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc_obj.page}")

    doc.build(story, onFirstPage=_page_number, onLaterPages=_page_number)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def _load_batch_texts(uploads: list) -> list[tuple[str, str]]:
    batch: list[tuple[str, str]] = []

    for upload in uploads:
        name = getattr(upload, "name", "") or "upload.txt"
        lower = name.lower()

        if lower.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(upload.getvalue())) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    inner_name = info.filename
                    if not inner_name.lower().endswith(".txt"):
                        continue
                    data = zf.read(info)
                    batch.append((f"{name}::{Path(inner_name).name}", data.decode("utf-8", errors="replace")))
        else:
            batch.append((name, upload.getvalue().decode("utf-8", errors="replace")))

    return batch


def main() -> None:
    st.set_page_config(page_title="Microdrama Promo Extractor", page_icon="🎬", layout="wide")
    st.title("Microdrama Promo Extractor")
    st.caption("Upload a screenplay and extract promo clips.")

    env_api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("PROMO_EVAL_MODEL", "gemini-3.5-flash-lite")

    with st.sidebar:
        st.header("Input")
        st.caption("API key and model are loaded from `.env`.")

        with st.form("promo_form", clear_on_submit=False):
            uploads = st.file_uploader(
                "Upload screenplay .txt files or a .zip",
                type=["txt", "zip"],
                accept_multiple_files=True,
            )
            min_duration = st.slider("Minimum duration (sec)", min_value=10.0, max_value=40.0, value=20.0, step=0.5)
            max_duration = st.slider("Maximum duration (sec)", min_value=30.0, max_value=90.0, value=60.0, step=0.5)
            start = st.form_submit_button("Start")

    if not start:
        st.info("Upload a screenplay TXT file and click Start.")
        return

    if not uploads:
        st.warning("Upload one or more TXT files, or a ZIP containing TXT episodes.")
        return

    status = st.status("Finding promo clips...", expanded=True)
    with status:
        batch_inputs = _load_batch_texts(uploads)
        if not batch_inputs:
            status.update(label="No episode text found.", state="error", expanded=False)
            st.warning("No .txt episodes were found in the upload.")
            return

        parser = ScriptParser()
        generator = CandidateGenerator(min_duration=min_duration, max_duration=max_duration)
        all_candidates = []
        episode_count = 0

        status.write(f"Parsing {len(batch_inputs)} episode files...")
        for source_name, raw_text in batch_inputs:
            parsed = parser.parse_text(raw_text, source_name=source_name)
            episode_count += 1
            all_candidates.extend(generator.generate(parsed.blocks))

        if not all_candidates:
            status.update(label="No valid promo candidates found.", state="error", expanded=False)
            st.warning("No valid promo candidates were found with the current window settings.")
            return

        status.write(f"Scoring {len(all_candidates)} candidates with LLM...")
        for cand in all_candidates:
            cand.evaluation = evaluate_candidate(
                cand,
                api_key=env_api_key or None,
                model=model,
            )

        status.write("Ranking and deduplicating promos...")
        ranked = rank_promos(all_candidates, max_promos=5)
        status.update(label="Promo extraction complete.", state="complete", expanded=False)

    top_row = st.container(border=True)
    with top_row:
        a, b, c, d = st.columns(4)
        a.metric("Episodes", episode_count)
        b.metric("Global Candidates", len(all_candidates))
        c.metric("Top Promos", len(ranked))
        d.metric("Max Ranking", "5")

    st.subheader("Global Top 5 Promo Clips")
    if not ranked:
        st.warning("No promos survived ranking and overlap filtering.")
        return

    for idx, cand in enumerate(ranked, start=1):
        ev = cand.evaluation
        assert ev is not None

        with st.container(border=True):
            st.markdown(f"### Promo #{idx}")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Composite Score", f"{cand.composite_score:.2f}")
            m2.metric("Duration (s)", f"{cand.duration_sec:.1f}")
            m3.metric("Start → End", f"{cand.start_timestamp_fmt} → {cand.end_timestamp_fmt}")
            m4.metric("Episode", cand.episode_title or "Untitled")

            if cand.evaluation_status.startswith("Gemini failed:"):
                st.error(cand.evaluation_status)
                st.caption("Mock fallback used for scoring.")
            elif cand.evaluation_status == "Mock fallback":
                st.info(cand.evaluation_status)

            _render_score_breakdown(cand)

            st.markdown(
                f"<div style='margin-top:0.5rem; color:#16a34a;'><b>Hook:</b> {escape(ev.hook_element)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='margin-top:0.25rem; color:#dc2626;'><b>Cliffhanger:</b> {escape(ev.cliffhanger_element)}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Justification:** {escape(ev.justification)}")

            with st.expander("Full Promo Transcript", expanded=False):
                st.code(cand.full_transcript, language="text")

            with st.expander("Failure / Quality Check Flags", expanded=False):
                if cand.diagnostics:
                    for tag in cand.diagnostics:
                        st.warning(tag)
                else:
                    st.success("No automatic failure tags detected.")

    export_payload = {
        "batch_source": [name for name, _ in batch_inputs],
        "episodes": episode_count,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "promos": [promo.to_dict() for promo in ranked],
    }
    export_json = json.dumps(export_payload, ensure_ascii=False, indent=2)
    export_pdf = _build_pdf_report(export_payload)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            label="Download results as JSON",
            data=export_json,
            file_name=f"{_safe_filename('batch')}_promos.json",
            mime="application/json",
        )
    with c2:
        st.download_button(
            label="Download report as PDF",
            data=export_pdf,
            file_name=f"{_safe_filename('batch')}_promos.pdf",
            mime="application/pdf",
        )

    with st.expander("Raw JSON Preview", expanded=False):
        st.code(export_json, language="json")


if __name__ == "__main__":
    main()
