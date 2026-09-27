"""Outputs that show the relationship: summary -> peak points -> time index -> clips."""
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.transcribe import fmt_time


def save_peaks_csv(peaks, path):
    cols = ["#", "start", "end", "clip_file", "score", "text_score", "audio_score",
            "visual_score", "reason", "dialogue"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, p in enumerate(peaks, 1):
            w.writerow([i, fmt_time(p["start"]), fmt_time(p["end"]), p.get("clip_file", ""),
                        p["score"], p["text_score"], p["audio_score"], p["visual_score"],
                        p["reason"], p["text"][:200]])


def save_mapping_csv(mapping, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["summary_sentence", "time_start", "time_end", "similarity", "sentence"])
        for m in mapping:
            w.writerow([m["sentence_id"] + 1, fmt_time(m["window_start"]),
                        fmt_time(m["window_end"]), m["similarity"], m["sentence"]])


def plot_timeline(windows, peaks, path):
    mids = [(w["start"] + w["end"]) / 2 / 60 for w in windows]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(mids, [w["text_score"] for w in windows], label="text", alpha=0.6)
    ax.plot(mids, [w["audio_score"] for w in windows], label="audio", alpha=0.6)
    ax.plot(mids, [w["visual_score"] for w in windows], label="visual", alpha=0.6)
    ax.plot(mids, [w["score"] for w in windows], label="fused", color="black", lw=2)
    for p in peaks:
        ax.axvspan(p["start"] / 60, p["end"] / 60, color="red", alpha=0.18)
    ax.set_xlabel("Film time (minutes)")
    ax.set_ylabel("Normalised score")
    ax.set_title("Multimodal importance over time (red = segments used in recap)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_markdown_report(path, info, story, mapping, peaks, evaluation, recap_secs):
    lines = ["# Movie Recap – Run Report", "",
             f"- Input duration: {fmt_time(info['duration'])}",
             f"- Recap duration: {fmt_time(recap_secs)} ({recap_secs / 60:.2f} min)",
             f"- Peak points used: {len(peaks)}", "",
             "## Summary story", "", story, "",
             "## Summary sentence → time index", "",
             "| # | Time | Similarity | Sentence |", "|---|---|---|---|"]
    for m in mapping:
        lines.append(f"| {m['sentence_id'] + 1} | {fmt_time(m['window_start'])}–"
                     f"{fmt_time(m['window_end'])} | {m['similarity']} | {m['sentence']} |")
    lines += ["", "## Peak points (recap order)", "",
              "| # | Time | Fused | Text | Audio | Visual | Reason |", "|---|---|---|---|---|---|---|"]
    for i, p in enumerate(peaks, 1):
        lines.append(f"| {i} | {fmt_time(p['start'])}–{fmt_time(p['end'])} | {p['score']} | "
                     f"{p['text_score']} | {p['audio_score']} | {p['visual_score']} | {p['reason']} |")
    if evaluation:
        lines += ["", "## Evaluation", "", "```", json.dumps(evaluation, indent=2), "```"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
