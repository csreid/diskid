import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from diskid.stt import RemoteTranscriber, TranscriptionError


class Handler(BaseHTTPRequestHandler):
    seen: list = []
    reply = (200, json.dumps({"text": "  hello there \n"}).encode())

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        Handler.seen.append((self.path, dict(self.headers), body))
        code, payload = Handler.reply
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    Handler.seen = []
    Handler.reply = (200, json.dumps({"text": "  hello there \n"}).encode())
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture
def wav(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFFfakewavdata")
    return p


def test_request_shape_and_response(server, wav):
    t = RemoteTranscriber(server + "/", "Systran/faster-whisper-small", api_key="sekret")
    assert t.transcribe(wav, language="en") == "hello there"
    path, headers, body = Handler.seen[0]
    assert path == "/v1/audio/transcriptions"
    assert headers["Authorization"] == "Bearer sekret"
    assert headers["Content-Type"].startswith("multipart/form-data; boundary=")
    assert b'name="model"\r\n\r\nSystran/faster-whisper-small' in body
    assert b'name="language"\r\n\r\nen' in body
    assert b'name="response_format"\r\n\r\njson' in body
    assert b'filename="clip.wav"' in body and b"RIFFfakewavdata" in body


def test_omits_language_and_auth_when_unset(server, wav):
    RemoteTranscriber(server, "m").transcribe(wav)
    _, headers, body = Handler.seen[0]
    assert "Authorization" not in headers and b'name="language"' not in body


def test_http_error_raises_transcription_error(server, wav):
    Handler.reply = (404, b'{"detail": "model not found"}')
    with pytest.raises(TranscriptionError, match="404.*model not found"):
        RemoteTranscriber(server, "nope").transcribe(wav)


def test_bad_payload_raises_transcription_error(server, wav):
    Handler.reply = (200, b"<html>proxy page</html>")
    with pytest.raises(TranscriptionError, match="unexpected"):
        RemoteTranscriber(server, "m").transcribe(wav)


def test_connection_refused_is_oserror(wav):
    with pytest.raises(OSError):
        RemoteTranscriber("http://127.0.0.1:1/v1", "m", timeout=2).transcribe(wav)
