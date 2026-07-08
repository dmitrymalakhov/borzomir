from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from .config import Settings


class SpeechError(RuntimeError):
    """Raised when local speech recognition or synthesis fails."""


class LocalSpeechService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def transcribe_telegram_voice(self, voice_data: bytes) -> str:
        with tempfile.TemporaryDirectory(prefix="borzomir-stt-") as directory_name:
            directory = Path(directory_name)
            input_path = directory / "voice.ogg"
            wav_path = directory / "voice.wav"
            transcript_base = directory / "transcript"
            transcript_path = transcript_base.with_suffix(".txt")

            input_path.write_bytes(voice_data)
            self._run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(input_path),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    str(wav_path),
                ]
            )
            result = self._run(
                [
                    self.settings.whisper_bin,
                    "-m",
                    self.settings.whisper_model_path,
                    "-f",
                    str(wav_path),
                    "-l",
                    self.settings.whisper_language,
                    "-nt",
                    "-otxt",
                    "-of",
                    str(transcript_base),
                ]
            )

            raw_text = transcript_path.read_text(encoding="utf-8", errors="replace") if transcript_path.exists() else result.stdout
            transcript = normalize_whisper_transcript(raw_text)
            if not transcript:
                raise SpeechError("Whisper вернул пустую расшифровку.")
            return transcript

    def synthesize_telegram_voice(self, text: str) -> bytes:
        return self._synthesize(text=text, output_name="answer.ogg", ffmpeg_args=["-c:a", "libopus", "-b:a", "32k", "-application", "voip"])

    def synthesize_telegram_audio(self, text: str) -> bytes:
        return self._synthesize(text=text, output_name="answer.mp3", ffmpeg_args=["-c:a", "libmp3lame", "-b:a", "64k"])

    def _synthesize(self, *, text: str, output_name: str, ffmpeg_args: list[str]) -> bytes:
        with tempfile.TemporaryDirectory(prefix="borzomir-tts-") as directory_name:
            directory = Path(directory_name)
            wav_path = directory / "answer.wav"
            output_path = directory / output_name

            command = [
                self.settings.piper_bin,
                "--model",
                self.settings.piper_model_path,
                "--config",
                self.settings.piper_config_path,
                "--output_file",
                str(wav_path),
            ]
            self._run(command, input_text=normalize_tts_text(text))
            self._run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(wav_path),
                    *ffmpeg_args,
                    str(output_path),
                ]
            )
            if not output_path.exists():
                raise SpeechError(f"Piper не создал файл {output_name} для Telegram.")
            return output_path.read_bytes()

    def _run(self, command: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(
                command,
                input=input_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.settings.speech_timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SpeechError(f"Не найден локальный бинарник: {command[0]}") from exc
        except subprocess.TimeoutExpired as exc:
            raise SpeechError(f"Локальная speech-команда не уложилась в таймаут: {command[0]}") from exc

        if result.returncode != 0:
            details = "\n".join(part.strip() for part in (result.stderr, result.stdout) if part.strip())
            raise SpeechError(f"Локальная speech-команда завершилась с ошибкой: {command[0]}\n{details[:1200]}")
        return result


def normalize_whisper_transcript(raw_text: str) -> str:
    lines: list[str] = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\[[^\]]+\]\s*", "", line).strip()
        if not cleaned or cleaned in {"[BLANK_AUDIO]", "(blank_audio)"}:
            continue
        lines.append(cleaned)
    return " ".join(lines).strip()


def normalize_tts_text(text: str) -> str:
    cleaned = re.sub(r"```.*?```", " фрагмент кода ", text, flags=re.DOTALL)
    cleaned = re.sub(
        r"^\s*\[?\s*(голосовое\s+сообщение|voice\s+message|аудио(?:сообщение)?)"
        r"(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\s*\]?\s*[:—-]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"https?://\S+", " ссылка ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
