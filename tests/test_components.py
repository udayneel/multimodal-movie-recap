"""
Component tests (guideline 15): each stage is tested in isolation on a small
synthetic video before the full pipeline is run.  Run with:  pytest -q
"""
import os
import subprocess
import numpy as np
import pytest

from src.media import extract_audio, video_info
from src import peaks as pk
from src.recap import build_recap, duration_of
from src.evaluate import bleu_scores, evaluate
from src.summarize import split_sentences, chunk_words

TMP = "tests/_tmp"


@pytest.fixture(scope="module")
def synthetic_video():
    """60 s test video: static first half, moving pattern second half,
    quiet tone with a loud burst at 40-45 s."""
    os.makedirs(TMP, exist_ok=True)
    path = os.path.join(TMP, "synthetic.mp4")
    if not os.path.exists(path):
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=gray:s=320x240:d=30:r=25",
            "-f", "lavfi", "-i", "testsrc=s=320x240:d=30:r=25",
            "-f", "lavfi", "-i", "sine=f=440:d=60",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1[v];"
            "[2:a]volume='if(between(t,40,45),1.0,0.05)':eval=frame[a]",
            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", path], check=True)
    return path


@pytest.fixture(scope="module")
def segments():
    lines = ["The family arrives at the old farmhouse.", "They talk about the weather.",
             "A stranger attacks them in the cemetery.", "Barbara runs away screaming.",
             "Ben boards up the windows of the house.", "The radio announces the dead are rising."]
    return [{"start": i * 10.0, "end": i * 10.0 + 9.0, "text": t} for i, t in enumerate(lines)]


def test_video_info(synthetic_video):
    info = video_info(synthetic_video)
    assert info["width"] == 320 and abs(info["duration"] - 60) < 1.5


def test_audio_extraction(synthetic_video):
    wav = extract_audio(synthetic_video, os.path.join(TMP, "a.wav"))
    assert os.path.getsize(wav) > 16000 * 2 * 50   # ~60 s of 16-bit mono


def test_windows(segments):
    w = pk.build_windows(segments, 20)
    assert len(w) == 2 and w[0]["start"] == 0 and w[-1]["end"] == 59


def test_audio_score_finds_loud_part(synthetic_video, segments):
    wav = os.path.join(TMP, "a.wav")
    w = pk.build_windows(segments, 5)          # one window per segment
    a = pk.audio_scores(w, wav)
    assert int(np.argmax(a)) == 4               # segment 40-49 s holds the burst


def test_visual_score_finds_motion(synthetic_video, segments):
    w = pk.build_windows(segments, 5)
    t, d = pk.visual_activity(synthetic_video, 2)
    v = pk.visual_scores(w, t, d)
    assert v[3:].mean() > v[:3].mean()          # moving half > static half


def test_text_score_matches_sentence(segments):
    w = pk.build_windows(segments, 5)
    sents = ["Ben boards up the house windows.", "A stranger attacks in the cemetery."]
    t, sim = pk.text_scores(w, sents, "not-a-real-model")   # forces TF-IDF fallback
    assert int(np.argmax(sim[0])) == 4 and int(np.argmax(sim[1])) == 2


def test_selection_respects_budget(segments):
    w = pk.build_windows(segments, 5)
    sents = ["Ben boards up the windows.", "A stranger attacks in the cemetery."]
    t, sim = pk.text_scores(w, sents, "not-a-real-model")
    pk.fuse(w, t, np.zeros(len(w)), np.zeros(len(w)), {"text": 1, "audio": 0, "visual": 0})
    peaks, mapping = pk.select_peaks(w, sim, sents, budget=25, padding=0.5)
    assert sum(p["clip_end"] - p["clip_start"] for p in peaks) <= 25
    assert [p["start"] for p in peaks] == sorted(p["start"] for p in peaks)
    assert {m["window_id"] for m in mapping} <= {p["id"] for p in peaks}


def test_recap_build(synthetic_video, segments):
    wav = os.path.join(TMP, "a.wav")
    peaks = [{"clip_start": 5, "clip_end": 10}, {"clip_start": 40, "clip_end": 46}]
    out = build_recap(synthetic_video, wav, peaks, os.path.join(TMP, "clips"),
                      os.path.join(TMP, "recap.mp4"))
    assert abs(duration_of(out) - 11) < 1.0
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                              "-of", "csv=p=0", out], capture_output=True, text=True).stdout
    assert "audio" in streams and "video" in streams


def test_bleu():
    ref = "the dead rise and attack the people hiding in the farmhouse"
    assert bleu_scores(ref, ref)["BLEU-4"] > 0.99
    assert bleu_scores("a cat sat on a mat", ref)["BLEU-4"] < 0.1
    assert "rouge" in evaluate(ref, ref)


def test_text_helpers():
    assert len(split_sentences("Ben runs away. He hides. The zombies come at night!")) == 2
    assert len(chunk_words("a " * 250, 100)) == 3
