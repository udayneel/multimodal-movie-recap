"""Steps 5-6 – Multimodal peak-point identification and timestamp mapping.

Each ~30 s window of the film gets three scores:
  text   : how semantically close its dialogue is to the summary story
  audio  : loudness (RMS energy) – shouts, music swells, gunshots
  visual : frame-to-frame change measured with OpenCV – action, cuts, motion
The scores are min-max normalised and fused with configurable weights.
"""
import numpy as np
import cv2
from scipy.io import wavfile


# ---------------------------------------------------------------- windows
def build_windows(segments: list, window_seconds: float = 30) -> list:
    """Group consecutive transcript segments into windows of roughly `window_seconds`."""
    windows, cur = [], []
    for seg in segments:
        cur.append(seg)
        if cur[-1]["end"] - cur[0]["start"] >= window_seconds:
            windows.append(_make_window(cur))
            cur = []
    if cur:
        windows.append(_make_window(cur))
    for i, w in enumerate(windows):
        w["id"] = i
    return windows


def _make_window(segs):
    return {"start": segs[0]["start"], "end": segs[-1]["end"],
            "text": " ".join(s["text"] for s in segs)}


def minmax(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min() if len(x) else 0
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


# ---------------------------------------------------------------- text
def _embed(texts: list, model_name: str) -> np.ndarray:
    """Sentence-transformer embeddings; falls back to TF-IDF if the model is unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return model.encode(texts, normalize_embeddings=True)
    except Exception as e:  # offline / not installed
        print(f"[peaks] sentence-transformers unavailable ({type(e).__name__}); using TF-IDF")
        from sklearn.feature_extraction.text import TfidfVectorizer
        m = TfidfVectorizer(stop_words="english").fit_transform(texts).toarray()
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)


def text_scores(windows: list, summary_sents: list, model_name: str):
    """Returns (score per window, similarity matrix [sentence x window])."""
    emb = _embed(summary_sents + [w["text"] for w in windows], model_name)
    s_emb, w_emb = emb[:len(summary_sents)], emb[len(summary_sents):]
    sim = s_emb @ w_emb.T
    return sim.max(axis=0), sim


# ---------------------------------------------------------------- audio
def audio_scores(windows: list, wav_path: str) -> np.ndarray:
    sr, audio = wavfile.read(wav_path)
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio /= (np.abs(audio).max() + 1e-9)
    out = []
    for w in windows:
        seg = audio[int(w["start"] * sr):int(w["end"] * sr)]
        if len(seg) == 0:
            out.append(0.0)
            continue
        # 95th percentile of short-frame RMS captures peaks better than the mean
        frame = sr // 10
        n = max(1, len(seg) // frame)
        rms = np.sqrt(np.mean(seg[:n * frame].reshape(n, -1) ** 2, axis=1)) if len(seg) >= frame \
            else np.array([np.sqrt(np.mean(seg ** 2))])
        out.append(float(np.percentile(rms, 95)))
    return np.array(out)


# ---------------------------------------------------------------- visual
def visual_activity(video_path: str, sample_fps: float = 2) -> tuple:
    """Sample frames with OpenCV and measure mean absolute difference between samples."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    times, diffs, prev, idx = [], [], None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if prev is not None:
                times.append(idx / fps)
                diffs.append(float(np.mean(cv2.absdiff(gray, prev))))
            prev = gray
        idx += 1
    cap.release()
    return np.array(times), np.array(diffs)


def visual_scores(windows: list, times: np.ndarray, diffs: np.ndarray) -> np.ndarray:
    out = []
    for w in windows:
        mask = (times >= w["start"]) & (times <= w["end"])
        out.append(float(diffs[mask].mean()) if mask.any() else 0.0)
    return np.array(out)


# ---------------------------------------------------------------- fusion + selection
def fuse(windows, t, a, v, weights):
    t, a, v = minmax(t), minmax(a), minmax(v)
    fused = weights["text"] * t + weights["audio"] * a + weights["visual"] * v
    for i, w in enumerate(windows):
        w.update(text_score=round(float(t[i]), 3), audio_score=round(float(a[i]), 3),
                 visual_score=round(float(v[i]), 3), score=round(float(fused[i]), 3))
    return fused


def select_peaks(windows, sim, summary_sents, budget, padding=1.0):
    """
    1) Coverage: for every summary sentence, take the window that best matches it
       (this ties each part of the story to a real moment in the film).
    2) Fill: add the remaining highest-fused-score windows while the budget allows.
    Output is sorted chronologically so the recap follows the story order.
    """
    def dur(w):
        return (w["end"] - w["start"]) + 2 * padding

    chosen, total, mapping = {}, 0.0, []
    for si, sent in enumerate(summary_sents):
        order = np.argsort(-sim[si])
        best = int(order[0])
        mapping.append({"sentence_id": si, "sentence": sent, "window_id": best,
                        "similarity": round(float(sim[si, best]), 3)})
        if best not in chosen and total + dur(windows[best]) <= budget:
            chosen[best] = f"summary sentence {si + 1}"
            total += dur(windows[best])

    for i in np.argsort([-w["score"] for w in windows]):
        i = int(i)
        if i in chosen:
            continue
        if total + dur(windows[i]) > budget:
            continue
        chosen[i] = "high multimodal score"
        total += dur(windows[i])

    peaks = []
    for i in sorted(chosen, key=lambda k: windows[k]["start"]):
        p = dict(windows[i])
        p["reason"] = chosen[i]
        p["clip_start"] = max(0.0, p["start"] - padding)
        p["clip_end"] = p["end"] + padding
        peaks.append(p)
    for m in mapping:
        m["window_start"] = windows[m["window_id"]]["start"]
        m["window_end"] = windows[m["window_id"]]["end"]
    print(f"[peaks] {len(peaks)} peak points selected, recap ≈ {total / 60:.1f} min")
    return peaks, mapping
