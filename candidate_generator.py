from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from parser import ScriptBlock


@dataclass(slots=True)
class PromoCandidate:
    candidate_id: str
    source_file: str
    episode_title: str
    start_timestamp_fmt: str
    end_timestamp_fmt: str
    duration_sec: float
    first_3s_text: str
    last_5s_text: str
    full_transcript: str

    start_sec: float = 0.0
    end_sec: float = 0.0
    start_idx: int = 0
    end_idx: int = 0
    boundary_score: float = 0.0
    blocks: list[ScriptBlock] = field(default_factory=list)

    evaluation: Optional[object] = None
    evaluation_backend: str = ""
    evaluation_status: str = ""
    evaluation_error: str = ""
    composite_score: float = 0.0
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source_file": self.source_file,
            "episode_title": self.episode_title,
            "start_timestamp_fmt": self.start_timestamp_fmt,
            "end_timestamp_fmt": self.end_timestamp_fmt,
            "duration_sec": round(self.duration_sec, 2),
            "first_3s_text": self.first_3s_text,
            "last_5s_text": self.last_5s_text,
            "full_transcript": self.full_transcript,
            "composite_score": round(self.composite_score, 2),
            "diagnostics": self.diagnostics,
            "evaluation": self.evaluation.model_dump() if hasattr(self.evaluation, "model_dump") else None,
        }


class CandidateGenerator:
    """Sliding-window generation with deterministic time math."""

    CAMERA_TERMS = (
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
        "overhead",
        "slowly",
        "camera",
    )

    def __init__(self, min_duration: float = 20.0, max_duration: float = 60.0):
        self.min_duration = float(min_duration)
        self.max_duration = float(max_duration)

    def generate(self, blocks: Sequence[ScriptBlock]) -> list[PromoCandidate]:
        ordered = sorted(blocks, key=lambda b: (b.start_sec, b.end_sec))
        raw_candidates: list[PromoCandidate] = []

        for i, start_block in enumerate(ordered):
            if not self._is_valid_start(start_block):
                continue

            for j in range(i, len(ordered)):
                end_block = ordered[j]
                if not self._is_valid_end(end_block):
                    continue

                duration = end_block.end_sec - start_block.start_sec
                if duration < self.min_duration:
                    continue
                if duration > self.max_duration:
                    break

                raw_candidates.append(self._build_candidate(ordered[i : j + 1], i, j))

        return self._deduplicate(raw_candidates)

    def _build_candidate(self, window_blocks: Sequence[ScriptBlock], start_idx: int, end_idx: int) -> PromoCandidate:
        start_block = window_blocks[0]
        end_block = window_blocks[-1]

        start_fmt = self._fmt_timestamp(start_block.start_sec)
        end_fmt = self._fmt_timestamp(end_block.end_sec)
        duration_sec = round(end_block.end_sec - start_block.start_sec, 2)

        first_3s_text = self._snippet_text(
            window_blocks=window_blocks,
            interval_start=start_block.start_sec,
            interval_end=start_block.start_sec + 3.0,
        )
        last_5s_text = self._snippet_text(
            window_blocks=window_blocks,
            interval_start=max(start_block.start_sec, end_block.end_sec - 5.0),
            interval_end=end_block.end_sec,
        )

        candidate_id = f"scene{start_block.scene_num:02d}_{start_idx:03d}_{end_idx:03d}"

        return PromoCandidate(
            candidate_id=candidate_id,
            source_file=getattr(start_block, "source_file", ""),
            episode_title=getattr(start_block, "episode_title", f"Scene {start_block.scene_num}"),
            start_timestamp_fmt=start_fmt,
            end_timestamp_fmt=end_fmt,
            duration_sec=duration_sec,
            first_3s_text=first_3s_text,
            last_5s_text=last_5s_text,
            full_transcript=self._format_transcript(window_blocks),
            start_sec=start_block.start_sec,
            end_sec=end_block.end_sec,
            start_idx=start_idx,
            end_idx=end_idx,
            boundary_score=self._boundary_score(start_block, end_block),
            blocks=list(window_blocks),
        )

    def _is_valid_start(self, block: ScriptBlock) -> bool:
        if block.block_type == "DIALOGUE":
            return True
        if block.block_type == "ACTION":
            return not self._looks_like_camera_text(block.text)
        return False

    def _is_valid_end(self, block: ScriptBlock) -> bool:
        if block.block_type == "DIALOGUE":
            return True
        if block.block_type == "ACTION":
            return not self._looks_like_camera_text(block.text)
        return False

    def _looks_like_camera_text(self, text: str) -> bool:
        low = text.lower()
        return any(term in low for term in self.CAMERA_TERMS)

    def _boundary_score(self, start_block: ScriptBlock, end_block: ScriptBlock) -> float:
        score = 0.0
        score += 2.0 if start_block.block_type == "DIALOGUE" else 1.0
        score += 1.5 if end_block.block_type == "DIALOGUE" else 1.0
        if not self._looks_like_camera_text(start_block.text):
            score += 0.25
        if not self._looks_like_camera_text(end_block.text):
            score += 0.25
        return round(score, 2)

    def _snippet_text(self, window_blocks: Sequence[ScriptBlock], interval_start: float, interval_end: float) -> str:
        snippets: list[str] = []
        for block in window_blocks:
            if block.end_sec <= interval_start:
                continue
            if block.start_sec >= interval_end:
                continue
            snippets.append(self._format_block_line(block))

        if not snippets and window_blocks:
            snippets.append(self._format_block_line(window_blocks[0]))

        return "\n".join(snippets)

    def _format_transcript(self, window_blocks: Sequence[ScriptBlock]) -> str:
        return "\n\n".join(self._format_block_line(block) for block in window_blocks)

    def _format_block_line(self, block: ScriptBlock) -> str:
        if block.block_type == "DIALOGUE":
            speaker = block.speaker or "SPEAKER"
            if block.tone:
                speaker = f"{speaker} ({block.tone})"
        else:
            speaker = block.block_type

        return f"[{self._fmt_timestamp(block.start_sec)} -> {self._fmt_timestamp(block.end_sec)}] {block.block_type} | {speaker}: {block.text}"

    def _fmt_timestamp(self, seconds: float) -> str:
        total_centis = int(round(seconds * 100))
        minutes, rem = divmod(total_centis, 6000)
        sec = rem / 100.0
        return f"{minutes:02d}:{sec:05.2f}"

    def _deduplicate(self, candidates: Sequence[PromoCandidate]) -> list[PromoCandidate]:
        ordered = sorted(candidates, key=lambda c: (-c.boundary_score, c.duration_sec, c.start_sec, c.end_sec))
        kept: list[PromoCandidate] = []

        for cand in ordered:
            if all(self._overlap_ratio(cand, existing) <= 0.80 for existing in kept):
                kept.append(cand)

        return sorted(kept, key=lambda c: (c.start_sec, c.end_sec))

    def _overlap_ratio(self, a: PromoCandidate, b: PromoCandidate) -> float:
        overlap = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
        if overlap <= 0:
            return 0.0
        smallest = min(a.duration_sec, b.duration_sec)
        return overlap / smallest if smallest > 0 else 0.0
