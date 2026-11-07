from __future__ import annotations

import json
import math
import os
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from candidate_generator import PromoCandidate

try:
    from google import genai
except Exception:  # pragma: no cover
    genai = None


HOOK_TYPES = [
    "slap_violence",
    "shocking_line",
    "reveal",
    "accusation_conflict",
    "threat",
    "humiliation",
    "desire_romance",
    "none",
]

CLIFF_TYPES = [
    "reveal_cut",
    "threat_cut",
    "decision_cut",
    "reversal_cut",
    "arrival_cut",
    "question_cut",
    "none",
]


class PromoEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    viewer_question: str = Field(
        "", description="The ONE concrete question a cold viewer is desperate to answer after the last frame"
    )
    hook_type: str = Field("none", description="Category of the hook")
    cliffhanger_type: str = Field("none", description="Category of the cliffhanger")
    resolution_leak: bool = Field(False, description="True if the clip itself resolves/answers the tension before it ends")

    hook_interrupt: Optional[int] = Field(None, ge=1, le=10)
    hook_curiosity: Optional[int] = Field(None, ge=1, le=10)
    hook_emotion: Optional[int] = Field(None, ge=1, le=10)
    cliff_open_loop: Optional[int] = Field(None, ge=1, le=10)
    cliff_stakes: Optional[int] = Field(None, ge=1, le=10)
    cliff_cut_timing: Optional[int] = Field(None, ge=1, le=10)

    hook_score: int = Field(..., ge=1, le=10)
    cliffhanger_score: int = Field(..., ge=1, le=10)
    clarity_score: int = Field(..., ge=1, le=10)
    hook_element: str = Field(...)
    cliffhanger_element: str = Field(...)
    justification: str = Field(...)
    red_flags: list[str] = Field(default_factory=list)


PROMO_EVALUATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "viewer_question": {"type": "string"},
        "hook_type": {"type": "string", "enum": HOOK_TYPES},
        "cliffhanger_type": {"type": "string", "enum": CLIFF_TYPES},
        "resolution_leak": {"type": "boolean"},
        "hook_interrupt": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "hook_curiosity": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "hook_emotion": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "cliff_open_loop": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "cliff_stakes": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "cliff_cut_timing": {"type": ["integer", "null"], "minimum": 1, "maximum": 10},
        "hook_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "cliffhanger_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "clarity_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "hook_element": {"type": "string"},
        "cliffhanger_element": {"type": "string"},
        "justification": {"type": "string"},
        "red_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "viewer_question",
        "hook_type",
        "cliffhanger_type",
        "resolution_leak",
        "hook_interrupt",
        "hook_curiosity",
        "hook_emotion",
        "cliff_open_loop",
        "cliff_stakes",
        "cliff_cut_timing",
        "hook_score",
        "cliffhanger_score",
        "clarity_score",
        "hook_element",
        "cliffhanger_element",
        "justification",
        "red_flags",
    ],
}


SYSTEM_PROMPT_TEMPLATE = """You are a veteran short-form drama editor who also understands attention psychology. You pick promo clips (ads) for vertical microdrama series on TikTok, Instagram Reels and YouTube Shorts.

The viewer is a stranger, thumb on the screen, ready to swipe in under 2 seconds. A great promo does two things: (1) it STOPS the thumb, and (2) after the last frame it leaves a question so itchy that the viewer opens the app to scratch it. You are not judging writing quality. You are judging stopping power and pull.

=== HOOK: judge ONLY FIRST_3S_TEXT ===
Use these lenses:
1. Pattern interrupt (orienting response): a sudden violation of expectation, such as a slap, a shout, a door kicked open, a shocking line. Greetings, scene-setting, walking, thinking, small talk = no interrupt.
2. Curiosity gap (Loewenstein): the viewer instantly knows a little and feels a specific hole in what they know. "Who is she to him?" works. Total confusion does not, and neither does full information.
3. Emotional and moral charge: injustice, humiliation, betrayal, desire, fear. The power imbalance must be readable in seconds (who is the underdog, who is the aggressor). Threats and wrongs grab attention faster than pleasantries (negativity bias).
4. In medias res: it starts at the peak of a conflict, not the build-up to it.
Score hook_interrupt, hook_curiosity, hook_emotion separately, then hook_score overall.

=== CLIFFHANGER: judge LAST_5S_TEXT (use NEXT_AFTER_CLIP only to see where the cut falls) ===
Use these lenses:
1. Open loop (Zeigarnik effect, need for closure): the ending plants ONE specific question in the viewer's head. Write it in viewer_question. If you cannot phrase it as a concrete question, the cliffhanger is weak.
2. Cut after the turn, before the payoff: the best cut lands on a reversal or recognition (Aristotle's peripeteia and anagnorisis), or on a threat or decision at its moment of maximum uncertainty, right before the consequence. Set resolution_leak = true if the clip itself answers the question or wraps the conflict up.
3. Stakes and loss aversion: something precious (love, life, a secret, status, a child) is visibly about to be lost or exposed.
4. Dramatic irony: the viewer knows something a character does not (the bomb under the table).
5. Peak-end rule: the last seconds should be the emotional peak of the clip, not a decline from it.
Weak endings: reflection, exposition, goodbyes, a conflict that settles down, a confusing beat that needs earlier episodes, a mood with no question.
Score cliff_open_loop, cliff_stakes, cliff_cut_timing separately, then cliffhanger_score overall.

=== CLARITY ===
Could someone who never saw the show understand who the people are, what they want and why it matters, using only this clip? Heavy reliance on earlier episodes, too many characters, or long camera or action-only passages lower clarity.

=== CALIBRATION (important) ===
1-3 = weak or absent, 4-6 = average (most clips belong here), 7-8 = strong, 9-10 = rare (you would stop scrolling for it). If you give 7 or above to everything you are being too generous. Be a harsh editor.

=== RULES ===
- Use exact lines or visual actions already present in the transcript for hook_element and cliffhanger_element.
- Do not do timestamp or duration math.
- red_flags: short phrases for real weaknesses (empty list if none).
- Write the reasoning fields (viewer_question, hook_type, cliffhanger_type, resolution_leak) before the scores.
- Return only valid JSON matching the required schema.

FIRST_3S_TEXT:
{first_3s_text}

LAST_5S_TEXT:
{last_5s_text}

NEXT_AFTER_CLIP (not part of the clip; may be empty):
{next_after_clip}

FULL_TRANSCRIPT:
{full_transcript}
"""


def evaluate_candidate(
    candidate: PromoCandidate,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> PromoEvaluation:
    if not api_key:
        candidate.evaluation_backend = "mock"
        candidate.evaluation_status = "Mock fallback"
        candidate.evaluation_error = ""
        return mock_evaluate_candidate(candidate)

    model = model or os.getenv("PROMO_EVAL_MODEL", "gemini-3.5-flash-lite")

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        first_3s_text=candidate.first_3s_text,
        last_5s_text=candidate.last_5s_text,
        next_after_clip=getattr(candidate, "after_end_text", "") or "(not provided)",
        full_transcript=candidate.full_transcript,
    )

    if genai is not None:
        try:
            evaluation = _evaluate_with_gemini(prompt=prompt, api_key=api_key, model=model)
            evaluation = apply_caps(evaluation)
            candidate.evaluation_backend = "gemini"
            candidate.evaluation_status = "Gemini OK"
            candidate.evaluation_error = ""
            return evaluation
        except Exception as exc:
            candidate.evaluation_backend = "mock"
            candidate.evaluation_status = f"Gemini failed: {exc}"
            candidate.evaluation_error = str(exc)
            return mock_evaluate_candidate(candidate)

    candidate.evaluation_backend = "mock"
    candidate.evaluation_status = "Mock fallback"
    candidate.evaluation_error = ""
    return mock_evaluate_candidate(candidate)


def promo_score(ev: PromoEvaluation) -> float:
    w_hook, w_cliff, w_clarity = 0.40, 0.35, 0.25
    return ev.hook_score ** w_hook * ev.cliffhanger_score ** w_cliff * ev.clarity_score ** w_clarity


def is_postable(ev: PromoEvaluation) -> bool:
    return ev.hook_score >= 6 and ev.cliffhanger_score >= 6 and ev.clarity_score >= 5 and not ev.resolution_leak


def apply_caps(ev: PromoEvaluation) -> PromoEvaluation:
    hook = ev.hook_score
    cliff = ev.cliffhanger_score

    hook_interrupt = ev.hook_interrupt if ev.hook_interrupt is not None else 1
    hook_curiosity = ev.hook_curiosity if ev.hook_curiosity is not None else 1

    if hook_interrupt <= 3 and hook_curiosity <= 3:
        hook = min(hook, 4)

    hook_subs = [s for s in (ev.hook_interrupt, ev.hook_curiosity, ev.hook_emotion) if s is not None]
    if hook_subs:
        hook = min(hook, math.ceil(sum(hook_subs) / len(hook_subs)) + 1)

    if ev.resolution_leak:
        cliff = min(cliff, 3)
    if len(ev.viewer_question.strip()) < 8 or ev.cliffhanger_type == "none":
        cliff = min(cliff, 4)

    cliff_subs = [s for s in (ev.cliff_open_loop, ev.cliff_stakes, ev.cliff_cut_timing) if s is not None]
    if cliff_subs:
        cliff = min(cliff, math.ceil(sum(cliff_subs) / len(cliff_subs)) + 1)

    return ev.model_copy(update={"hook_score": _clamp(1, 10, hook), "cliffhanger_score": _clamp(1, 10, cliff)})


def _evaluate_with_gemini(prompt: str, api_key: str, model: str) -> PromoEvaluation:
    if genai is None:
        raise RuntimeError("google-genai package is not installed")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
            "response_json_schema": PROMO_EVALUATION_JSON_SCHEMA,
        },
    )

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return PromoEvaluation.model_validate(_normalize_payload(parsed))

    content = getattr(response, "text", "") or ""
    return PromoEvaluation.model_validate(_normalize_payload(_extract_json(content)))


HOOK_WORDS = {
    "you",
    "no",
    "stop",
    "now",
    "wait",
    "betray",
    "betrayed",
    "kill",
    "dead",
    "never",
    "what",
    "why",
    "confess",
    "secret",
    "run",
    "gun",
    "threat",
    "forbidden",
    "slap",
    "slaps",
    "scream",
    "screams",
    "shove",
    "grabs",
    "blood",
    "smash",
    "slam",
    "liar",
    "lied",
    "divorce",
    "pregnant",
    "fired",
    "get out",
    "how dare",
}

CLIFF_WORDS = {
    "wait",
    "stop",
    "what",
    "no",
    "please",
    "don't",
    "cannot",
    "can't",
    "secret",
    "kill",
    "betray",
    "if you",
    "or else",
    "not yet",
    "who",
    "why",
    "unknown",
    "murder",
    "gone",
    "pregnant",
    "alive",
    "truth",
    "father",
    "baby",
    "open the door",
}

RESOLUTION_WORDS = {
    "thank you",
    "thanks",
    "okay",
    "ok",
    "goodbye",
    "bye",
    "i'm sorry",
    "forgive",
    "it's over",
    "i love you",
    "all right",
    "fine",
    "good night",
    "see you",
}

WARMUP_WORDS = {
    "good morning",
    "hello",
    "hi",
    "welcome",
    "how are you",
    "nice to meet",
    "thank you",
    "so,",
    "well,",
}


def mock_evaluate_candidate(candidate: PromoCandidate) -> PromoEvaluation:
    first = candidate.first_3s_text.strip()
    last = candidate.last_5s_text.strip()
    full = candidate.full_transcript.strip()

    first_words = len(first.split())
    hook_raw = (
        3
        + 1.0 * min(_keyword_hits(first, HOOK_WORDS), 3)
        + (1.5 if any(p in first for p in ("!", "?", "—")) else 0.0)
        + (1.0 if _has_direct_conflict(first) else 0.0)
        + (0.5 if 0 < first_words <= 12 else 0.0)
        - (1.5 if _keyword_hits(first, WARMUP_WORDS) else 0.0)
    )
    hook_score = _clamp(1, 10, round(hook_raw))

    cliff_raw = (
        3
        + 0.9 * min(_keyword_hits(last, CLIFF_WORDS), 3)
        + (2.0 if re.search(r"(\.\.\.|—|-$)", last) else 0.0)
        + (1.0 if re.search(r"[?!]\s*$", last) else 0.0)
        - (2.0 if _keyword_hits(last, RESOLUTION_WORDS) else 0.0)
    )
    cliffhanger_score = _clamp(1, 10, round(cliff_raw))

    clarity_score = _clamp(
        1,
        10,
        round(
            7
            - (1.5 if _camera_ratio(candidate) > 0.4 else 0.0)
            - (1.0 if _speaker_count(candidate) > 5 else 0.0)
            - (1.0 if len(full) > 1600 else 0.0)
        ),
    )

    hook_element = _best_line(first, HOOK_WORDS) or _first_non_empty_line(first) or candidate.first_3s_text[:120]
    cliffhanger_element = _best_line(last, CLIFF_WORDS) or _last_non_empty_line(last) or candidate.last_5s_text[:120]

    red_flags = []
    if hook_score <= 4:
        red_flags.append("weak or slow opening")
    if cliffhanger_score <= 4:
        red_flags.append("ending may resolve or lacks an open question")
    if clarity_score <= 5:
        red_flags.append("may need prior-episode context")

    justification = (
        f"Heuristic estimate: hook {hook_score}/10, cliffhanger {cliffhanger_score}/10, "
        f"clarity {clarity_score}/10 (keyword-based, not a real judgement)."
    )

    return PromoEvaluation(
        viewer_question="",
        hook_type="none",
        cliffhanger_type="none",
        resolution_leak=False,
        hook_interrupt=None,
        hook_curiosity=None,
        hook_emotion=None,
        cliff_open_loop=None,
        cliff_stakes=None,
        cliff_cut_timing=None,
        hook_score=hook_score,
        cliffhanger_score=cliffhanger_score,
        clarity_score=clarity_score,
        hook_element=hook_element,
        cliffhanger_element=cliffhanger_element,
        justification=justification,
        red_flags=red_flags,
    )


def _extract_json(content: str) -> dict:
    raw = content.strip()
    if not raw:
        raise ValueError("Empty LLM response")

    try:
        return json.loads(raw)
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return json.loads(raw[start : end + 1])

    raise ValueError(f"Could not parse JSON from response: {content[:300]!r}")


def _normalize_payload(payload: object) -> dict:
    if isinstance(payload, PromoEvaluation):
        return payload.model_dump()
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported response payload type: {type(payload)!r}")

    required = {
        "hook_score",
        "cliffhanger_score",
        "clarity_score",
        "hook_element",
        "cliffhanger_element",
        "justification",
        "viewer_question",
        "hook_type",
        "cliffhanger_type",
        "resolution_leak",
        "red_flags",
    }
    if required.issubset(payload.keys()):
        return {k: v for k, v in payload.items() if k in PromoEvaluation.model_fields}

    hook_eval = payload.get("hook_evaluation") or {}
    cliff_eval = payload.get("cliffhanger_evaluation") or {}
    clarity_eval = payload.get("clarity_evaluation") or {}

    def _score(value: object) -> int:
        raw = value.get("score", 1) if isinstance(value, dict) else value
        try:
            return _clamp(1, 10, int(raw))  # type: ignore[arg-type]
        except Exception:
            return 1

    def _text(value: object) -> str:
        if isinstance(value, dict):
            for key in ("reasoning", "text", "element", "value"):
                if value.get(key):
                    return str(value.get(key))
        return str(value) if value is not None else ""

    return {
        "viewer_question": str(payload.get("viewer_question") or ""),
        "hook_type": str(payload.get("hook_type") or "none"),
        "cliffhanger_type": str(payload.get("cliffhanger_type") or "none"),
        "resolution_leak": bool(payload.get("resolution_leak", False)),
        "hook_interrupt": _score(hook_eval),
        "hook_curiosity": _score(hook_eval),
        "hook_emotion": _score(hook_eval),
        "cliff_open_loop": _score(cliff_eval),
        "cliff_stakes": _score(cliff_eval),
        "cliff_cut_timing": _score(cliff_eval),
        "hook_score": _score(hook_eval),
        "cliffhanger_score": _score(cliff_eval),
        "clarity_score": _score(clarity_eval),
        "hook_element": _text(hook_eval) or str(payload.get("hook_element") or ""),
        "cliffhanger_element": _text(cliff_eval) or str(payload.get("cliffhanger_element") or ""),
        "justification": str(payload.get("pacing_and_flow") or payload.get("target_audience_fit") or payload.get("justification") or ""),
        "red_flags": list(payload.get("red_flags") or []),
    }


def _keyword_hits(text: str, keywords: set[str]) -> int:
    low = text.lower()
    return sum(1 for k in keywords if re.search(rf"(?<!\w){re.escape(k)}(?!\w)", low))


def _has_direct_conflict(text: str) -> bool:
    low = text.lower()
    return any(
        term in low
        for term in ("don't", "do not", "you lied", "liar", "leave", "back off", "move", "answer me", "get out", "how dare")
    )


def _speaker_count(candidate: PromoCandidate) -> int:
    speakers = set()
    for block in candidate.blocks:
        if block.speaker:
            speakers.add(block.speaker.strip().upper())
    return len(speakers)


def _camera_ratio(candidate: PromoCandidate) -> float:
    if candidate.duration_sec <= 0:
        return 0.0

    camera_dur = 0.0
    for block in candidate.blocks:
        txt = block.text.lower()
        if block.block_type == "CAMERA" or any(
            term in txt for term in ("pan", "zoom", "dolly", "tracking", "tilt", "slowly", "wide shot", "close-up")
        ):
            camera_dur += block.duration
    return camera_dur / candidate.duration_sec


def _best_line(text: str, keywords: set[str]) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    ranked = sorted(lines, key=lambda line: (_keyword_hits(line, keywords), len(line)), reverse=True)
    return ranked[0]


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _clamp(lo: int, hi: int, value: int) -> int:
    return max(lo, min(hi, value))
