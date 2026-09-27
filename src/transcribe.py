"""Step 3 – Speech-to-text with timestamps (faster-whisper)."""
import json


def transcribe(wav_path: str, model_size: str = "small", language: str = "en") -> list:
    """Return a list of segments: {start, end, text}. Timestamps are in seconds."""
    from faster_whisper import WhisperModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute)

    segments, info = model.transcribe(wav_path, language=language, vad_filter=True, beam_size=5)
    result = []
    for s in segments:
        text = s.text.strip()
        if text:
            result.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
    print(f"[transcribe] {len(result)} segments, audio {info.duration:.0f}s")
    return result


def save_transcript(segments: list, json_path: str, txt_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{fmt_time(s['start'])} - {fmt_time(s['end'])}] {s['text']}\n")


def load_transcript(json_path: str) -> list:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def fmt_time(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
