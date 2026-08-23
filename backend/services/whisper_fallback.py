from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def whisper_available() -> bool:
    return bool(shutil.which("yt-dlp"))


def transcribe_youtube(url: str, whisper_model: str = "small") -> str:
    """Download audio with yt-dlp and transcribe locally.

    Prefers mlx_whisper on Apple Silicon, then faster_whisper, then openai-whisper CLI.
    """
    if not shutil.which("yt-dlp"):
        raise ValueError("yt-dlp not installed — brew install yt-dlp")

    with tempfile.TemporaryDirectory(prefix="notionlearner_") as tmp:
        out_tmpl = str(Path(tmp) / "audio.%(ext)s")
        cmd = [
            "yt-dlp",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "-o",
            out_tmpl,
            "--no-playlist",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise ValueError(f"yt-dlp failed: {proc.stderr[-500:]}")

        audio_files = list(Path(tmp).glob("audio.*"))
        if not audio_files:
            raise ValueError("yt-dlp produced no audio file")
        audio_path = str(audio_files[0])

        # 1) mlx-whisper
        try:
            import mlx_whisper  # type: ignore

            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=f"mlx-community/whisper-{whisper_model}-mlx",
            )
            return _format_segments(result)
        except Exception:
            pass

        # 2) faster-whisper
        try:
            from faster_whisper import WhisperModel  # type: ignore

            model = WhisperModel(whisper_model, device="auto", compute_type="int8")
            segments, _info = model.transcribe(audio_path)
            lines = []
            for seg in segments:
                lines.append(f"[{_ts(seg.start)}] {seg.text.strip()}")
            if lines:
                return "\n".join(lines)
        except Exception:
            pass

        # 3) openai-whisper CLI
        if shutil.which("whisper"):
            proc = subprocess.run(
                [
                    "whisper",
                    audio_path,
                    "--model",
                    whisper_model,
                    "--output_format",
                    "txt",
                    "--output_dir",
                    tmp,
                ],
                capture_output=True,
                text=True,
            )
            txts = list(Path(tmp).glob("*.txt"))
            if txts:
                return txts[0].read_text(encoding="utf-8", errors="ignore")
            raise ValueError(f"whisper CLI failed: {proc.stderr[-500:]}")

        raise ValueError(
            "No Whisper backend found. Install one of: "
            "pip install mlx-whisper | faster-whisper, or brew install openai-whisper"
        )


def _ts(seconds: float) -> str:
    total = int(seconds or 0)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_segments(result: dict) -> str:
    segments = result.get("segments") or []
    if segments:
        lines = []
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if text:
                lines.append(f"[{_ts(seg.get('start', 0))}] {text}")
        if lines:
            return "\n".join(lines)
    return (result.get("text") or "").strip()
