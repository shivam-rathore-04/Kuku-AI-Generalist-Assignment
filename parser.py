from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class ScriptBlock:
    scene_num: int
    source_file: str
    episode_title: str
    block_type: str
    speaker: Optional[str]
    tone: Optional[str]
    start_sec: float
    end_sec: float
    duration: float
    text: str


@dataclass(slots=True)
class ParsedScript:
    title: str
    logline: str
    blocks: list[ScriptBlock] = field(default_factory=list)
    scene_count: int = 0
    total_duration_sec: float = 0.0
    source_name: str = ""


class ScriptParser:
    """Deterministic screenplay parser for microdrama promo extraction."""

    SCENE_RE = re.compile(r"^\s*SCENE\s+(?P<num>\d+)\s*$", re.IGNORECASE)
    TITLE_RE = re.compile(r"^\s*(?:Title|EPISODE NUMBER):\s*(?P<title>.+?)\s*$", re.IGNORECASE)
    LOGLINE_RE = re.compile(r"^\s*(?:Logline|EPISODE LOGLINE):\s*(?P<logline>.+?)\s*$", re.IGNORECASE)
    HEADER_RE = re.compile(r"^\s*\*\*(?P<label>.+?)\*\*\s*$")
    START_END_RE = re.compile(
        r"^\s*Start:\s*\((?P<start>[^)]+)\)\s*\|\s*End:\s*\((?P<end>[^)]+)\)\s*$",
        re.IGNORECASE,
    )
    SCENE_HEADING_RE = re.compile(r"^\s*Scene heading:\s*(?P<heading>.+?)\s*$", re.IGNORECASE)
    SCENE_START_RE = re.compile(r"^\s*Scene start:\s*(?P<start>.+?)\s*$", re.IGNORECASE)
    SCENE_END_RE = re.compile(r"^\s*Scene end:\s*(?P<end>.+?)\s*$", re.IGNORECASE)

    def timestamp_to_seconds(self, ts_str: str) -> float:
        """Convert HH:MM:SS.ms or MM:SS.ms timestamps to seconds."""
        ts = ts_str.strip().strip("()").strip()
        parts = [part.strip() for part in ts.split(":") if part.strip()]

        if not parts:
            raise ValueError(f"Invalid timestamp format: {ts_str!r}")

        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds

        raise ValueError(f"Invalid timestamp format: {ts_str!r}")

    def parse_file(self, file_path: str | Path) -> ParsedScript:
        path = Path(file_path)
        text = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_text(text, source_name=path.name)

    def parse_text(self, text: str, source_name: str = "") -> ParsedScript:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        title = Path(source_name).stem if source_name else "Untitled"
        logline = ""
        blocks: list[ScriptBlock] = []
        current_scene = 1
        seen_scenes: set[int] = set()
        last_scene_end = 0.0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line or line in {"--- SCRIPT START ---", "--- SCRIPT END ---"}:
                i += 1
                continue

            if m := self.TITLE_RE.match(line):
                title = m.group("title").strip()
                i += 1
                continue

            if m := self.LOGLINE_RE.match(line):
                logline = m.group("logline").strip()
                i += 1
                continue

            if m := self.SCENE_RE.match(line):
                current_scene = int(m.group("num"))
                seen_scenes.add(current_scene)
                i += 1
                continue

            if m := self.SCENE_END_RE.match(line):
                try:
                    last_scene_end = max(last_scene_end, self.timestamp_to_seconds(m.group("end")))
                except ValueError:
                    pass
                i += 1
                continue

            if m := self.HEADER_RE.match(line):
                label = m.group("label").strip()
                chunk: list[str] = []
                j = i + 1

                while j < len(lines):
                    nxt = lines[j].strip()
                    if self.SCENE_RE.match(nxt) or self.HEADER_RE.match(nxt):
                        break
                    chunk.append(lines[j])
                    j += 1

                block = self._parse_block(
                    label=label,
                    chunk_lines=chunk,
                    scene_num=current_scene,
                    episode_title=title,
                    source_file=source_name,
                )
                if block is not None:
                    blocks.append(block)
                    last_scene_end = max(last_scene_end, block.end_sec)

                i = j
                continue

            i += 1

        total_duration = last_scene_end if last_scene_end > 0 else max((b.end_sec for b in blocks), default=0.0)
        return ParsedScript(
            title=title,
            logline=logline,
            blocks=blocks,
            scene_count=len(seen_scenes),
            total_duration_sec=round(total_duration, 2),
            source_name=source_name,
        )

    def _parse_block(
        self,
        label: str,
        chunk_lines: list[str],
        scene_num: int,
        episode_title: str,
        source_file: str,
    ) -> Optional[ScriptBlock]:
        start_sec: Optional[float] = None
        end_sec: Optional[float] = None
        text_lines: list[str] = []

        for raw in chunk_lines:
            line = raw.strip()
            if not line or self.SCENE_HEADING_RE.match(line):
                continue

            if m := self.START_END_RE.match(line):
                start_sec = self.timestamp_to_seconds(m.group("start"))
                end_sec = self.timestamp_to_seconds(m.group("end"))
                continue

            if self.SCENE_START_RE.match(line) or self.SCENE_END_RE.match(line):
                continue

            text_lines.append(line)

        if start_sec is None or end_sec is None:
            return None

        block_type, speaker, tone = self._parse_label(label)
        text = self._normalize_text(text_lines)

        return ScriptBlock(
            scene_num=scene_num,
            source_file=source_file,
            episode_title=episode_title or "Untitled",
            block_type=block_type,
            speaker=speaker,
            tone=tone,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            duration=round(max(0.0, end_sec - start_sec), 2),
            text=text,
        )

    def _parse_label(self, label: str) -> tuple[str, Optional[str], Optional[str]]:
        clean = label.strip().rstrip(":").strip()
        upper = clean.upper()

        if upper in {"CAMERA", "ACTION"}:
            return upper, None, None

        speaker = clean
        tone = None
        match = re.match(r"^(?P<speaker>[^()]+?)\s*\((?P<tone>[^)]+)\)\s*$", clean)
        if match:
            speaker = match.group("speaker").strip()
            tone = match.group("tone").strip()

        return "DIALOGUE", speaker or None, tone

    def _normalize_text(self, lines: list[str]) -> str:
        return "\n".join(line.strip() for line in lines if line.strip())
