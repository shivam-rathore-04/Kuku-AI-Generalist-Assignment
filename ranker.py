from __future__ import annotations

import re

from candidate_generator import PromoCandidate
from evaluator import PromoEvaluation, is_postable, promo_score


def compute_composite_score(evaluation: PromoEvaluation) -> float:
    return round(promo_score(evaluation), 2)


def tag_failure_modes(candidate: PromoCandidate) -> list[str]:
    tags: list[str] = []

    if candidate.evaluation is None:
        return tags

    if _visual_pacing_risk(candidate) > 0.40:
        tags.append("TAG: VISUAL_PACING_RISK")

    if _mid_monologue_cut(candidate):
        tags.append("TAG: MID_MONOLOGUE_CUT")

    if candidate.evaluation.clarity_score < 5:
        tags.append("TAG: HIGH_EXPOSITION_DEPENDENCY")

    return tags


def rank_promos(candidates: list[PromoCandidate], max_promos: int = 3) -> list[PromoCandidate]:
    scored: list[PromoCandidate] = []
    fallback: list[PromoCandidate] = []

    for cand in candidates:
        if cand.evaluation is None:
            continue
        cand.composite_score = compute_composite_score(cand.evaluation)
        cand.diagnostics = tag_failure_modes(cand)
        fallback.append(cand)
        if is_postable(cand.evaluation):
            scored.append(cand)

    if not scored:
        scored = fallback

    scored.sort(key=lambda c: (c.composite_score, c.boundary_score, -c.duration_sec), reverse=True)

    selected: list[PromoCandidate] = []
    for cand in scored:
        if len(selected) >= max_promos:
            break
        if any(_overlap_seconds(cand, chosen) > 0.0 for chosen in selected):
            continue
        selected.append(cand)

    return selected


def _overlap_seconds(a: PromoCandidate, b: PromoCandidate) -> float:
    return max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))


def _visual_pacing_risk(candidate: PromoCandidate) -> float:
    if candidate.duration_sec <= 0:
        return 0.0

    camera_duration = 0.0
    for block in candidate.blocks:
        if _is_camera_like(block.text, block.block_type):
            camera_duration += block.duration

    return camera_duration / candidate.duration_sec


def _is_camera_like(text: str, block_type: str) -> bool:
    if block_type == "CAMERA":
        return True
    low = text.lower()
    return any(
        term in low
        for term in (
            "pan",
            "panning",
            "zoom",
            "push in",
            "pull back",
            "track",
            "tracking",
            "dolly",
            "tilt",
            "rack focus",
            "close-up",
            "wide shot",
            "slowly",
        )
    )


def _mid_monologue_cut(candidate: PromoCandidate) -> bool:
    dialogue_blocks = [b for b in candidate.blocks if b.block_type == "DIALOGUE" and b.text.strip()]
    if not dialogue_blocks:
        return False

    last = dialogue_blocks[-1].text.strip()
    if last.endswith(("...", "—", "-", ",")):
        return True
    if not re.search(r'[.!?]["\']?\s*$', last):
        if re.search(r"\b(and|but|because|so|or|if|when)\s*$", last, re.IGNORECASE):
            return True
        return True

    return False
