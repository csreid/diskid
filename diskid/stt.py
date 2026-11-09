"""Speech-to-text: local faster-whisper (GPU when available) or a remote server.

Two backends:

- local: `load_model` + `transcribe_wav`. Needs the optional `faster-whisper`
  dependency (`pip install 'diskid[local]'`); imported lazily so installs that
  only use a remote server stay light.
- remote: `RemoteTranscriber`, a client for any OpenAI-compatible
  `POST /v1/audio/transcriptions` server (Speaches, LocalAI, whisper.cpp's
  OpenAI-compat mode, ...). Stdlib only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


class Segment(NamedTuple):
    start: float
    end: float
    text: str


class TranscriptionError(RuntimeError):
    """The STT server answered with an error or an unusable response."""


def load_model(size: str = "small") -> tuple["WhisperModel", str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "local transcription needs faster-whisper: pip install 'diskid[local]' "
            "(or configure a remote STT server instead)"
        ) from e
    try:
        return WhisperModel(size, device="cuda", compute_type="float16"), "cuda"
    except Exception:
        return WhisperModel(size, device="cpu", compute_type="int8"), "cpu"


def transcribe_wav(
    model: "WhisperModel", wav: Path, language: str | None = None
) -> tuple[list[Segment], str]:
    """Transcribe a wav file; returns (segments, detected language)."""
    raw_segments, info = model.transcribe(str(wav), language=language)
    segments = [Segment(s.start, s.end, s.text.strip()) for s in raw_segments]
    return segments, info.language


class RemoteTranscriber:
    """Client for an OpenAI-compatible transcription endpoint.

    `base_url` is the API root, e.g. "http://speaches.svc:8000/v1" (the
    "/audio/transcriptions" path is appended). Network failures raise
    `OSError` subclasses; HTTP error responses raise `TranscriptionError`.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.url = base_url.rstrip("/") + "/audio/transcriptions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def transcribe(self, wav: Path, language: str | None = None) -> str:
        """Transcribe a wav file; returns the text (possibly empty)."""
        fields = {"model": self.model, "response_format": "json"}
        if language:
            fields["language"] = language
        body, content_type = _multipart(fields, "file", wav.name, wav.read_bytes())
        headers = {"Content-Type": content_type, "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read(500).decode("utf-8", "replace").strip()
            raise TranscriptionError(f"STT server returned HTTP {e.code}: {detail}") from e
        try:
            return str(json.loads(payload)["text"]).strip()
        except (ValueError, KeyError, TypeError) as e:
            raise TranscriptionError(
                f"unexpected STT response: {payload[:200]!r}") from e


def _multipart(
    fields: dict[str, str], file_field: str, filename: str, data: bytes,
) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
        + data + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
