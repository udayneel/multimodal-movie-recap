"""
Multimodal Movie Recap – end-to-end pipeline.

YouTube Video -> Audio + Video -> Transcript -> Summary Story -> Peak Points
-> Time Index -> Video Segment Extraction (OpenCV) -> Movie Recap (<10 min) -> BLEU
"""
import argparse
import json
import os

import config as cfg
from src.download import get_video, check_ffmpeg
from src.media import extract_audio, video_info, ensure_h264
from src.transcribe import transcribe, save_transcript, load_transcript
from src.summarize import summarize, split_sentences
from src import peaks as pk
from src.recap import build_recap, duration_of
from src.evaluate import evaluate
from src import report


def run(source, reference_path, out_dir):
    check_ffmpeg()
    if not source:
        raise SystemExit("Set a video: --source <YouTube URL or file> (or VIDEO_SOURCE env var)")
    os.makedirs(out_dir, exist_ok=True)
    p = lambda name: os.path.join(out_dir, name)

    # 1. Input video
    video = get_video(source, cfg.DATA_DIR)
    video = ensure_h264(video, cfg.DATA_DIR)
    info = video_info(video)
    print(f"[1] Video: {info['width']}x{info['height']} @ {info['fps']:.1f} fps, "
          f"{info['duration'] / 60:.1f} min")

    # 2. Audio + video separation
    wav = extract_audio(video, os.path.join(cfg.DATA_DIR, "audio.wav"))
    print(f"[2] Audio extracted -> {wav}")

    # 3. Speech-to-text (cached)
    segments = None
    if os.path.exists(p("transcript.json")):
        segments = load_transcript(p("transcript.json"))
        covered = segments[-1]["end"] if segments else 0
        if covered < 0.8 * info["duration"]:
            # e.g. transcript made from a half-written file – redo it
            print(f"[3] Cached transcript covers only {covered / 60:.1f} of "
                  f"{info['duration'] / 60:.1f} min – re-transcribing")
            segments = None
            for f in ("summary.json",):
                if os.path.exists(p(f)):
                    os.remove(p(f))
        else:
            print(f"[3] Loaded cached transcript ({len(segments)} segments, "
                  f"up to {covered / 60:.1f} min)")
    if segments is None:
        segments = transcribe(wav, cfg.WHISPER_MODEL, cfg.WHISPER_LANGUAGE)
        save_transcript(segments, p("transcript.json"), p("transcript.txt"))

    # 4. Summary story (cached)
    if os.path.exists(p("summary.json")):
        summ = json.load(open(p("summary.json"), encoding="utf-8"))
        print("[4] Loaded cached summary")
    else:
        summ = summarize(segments, cfg)
        json.dump(summ, open(p("summary.json"), "w", encoding="utf-8"), indent=2)
    story = summ["story"]
    open(p("summary_story.txt"), "w", encoding="utf-8").write(story)
    sents = split_sentences(story)

    # 5. Multimodal peak-point scoring
    windows = pk.build_windows(segments, cfg.WINDOW_SECONDS)
    t_score, sim = pk.text_scores(windows, sents, cfg.EMBED_MODEL)
    a_score = pk.audio_scores(windows, wav)
    times, diffs = pk.visual_activity(video, cfg.VISUAL_SAMPLE_FPS)
    v_score = pk.visual_scores(windows, times, diffs)
    pk.fuse(windows, t_score, a_score, v_score, cfg.WEIGHTS)
    print(f"[5] Scored {len(windows)} windows (text/audio/visual)")

    # 6. Peak selection + time-index mapping
    peaks, mapping = pk.select_peaks(windows, sim, sents, cfg.MAX_RECAP_SECONDS, cfg.CLIP_PADDING)

    # 7-8. Segment extraction (OpenCV) + recap assembly
    recap_path = p("movie_recap.mp4")
    build_recap(video, wav, peaks, cfg.CLIP_DIR if out_dir == cfg.OUTPUT_DIR
                else os.path.join(out_dir, "clips"), recap_path)
    recap_secs = duration_of(recap_path)
    status = "OK" if recap_secs < 600 else "OVER LIMIT"
    print(f"[8] Recap: {recap_secs / 60:.2f} min ({status}) -> {recap_path}")

    # 9. Evaluation
    evaluation = None
    if reference_path and os.path.exists(reference_path):
        reference = open(reference_path, encoding="utf-8").read()
        if reference.startswith("REPLACE THIS"):
            raise SystemExit("Write your reference summary in data/reference_summary.txt first")
        evaluation = evaluate(story, reference)
        json.dump(evaluation, open(p("evaluation.json"), "w"), indent=2)
        print(f"[9] BLEU: {evaluation['bleu']}  ROUGE: {evaluation['rouge']}")
    else:
        print("[9] No reference summary found – skipping BLEU")

    # Reports
    report.save_peaks_csv(peaks, p("peak_points.csv"))
    report.save_mapping_csv(mapping, p("summary_to_timestamps.csv"))
    report.plot_timeline(windows, peaks, p("importance_timeline.png"))
    report.write_markdown_report(p("report.md"), info, story, mapping, peaks, evaluation, recap_secs)
    json.dump(windows, open(p("windows_scored.json"), "w"), indent=2)
    print(f"Done. All outputs in {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multimodal movie recap generator")
    ap.add_argument("--source", default=cfg.VIDEO_SOURCE, help="YouTube URL or local video path")
    ap.add_argument("--reference", default=cfg.REFERENCE_SUMMARY, help="reference summary .txt for BLEU")
    ap.add_argument("--out", default=cfg.OUTPUT_DIR)
    args = ap.parse_args()
    run(args.source, args.reference, args.out)
