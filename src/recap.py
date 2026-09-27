"""Steps 7-8 – Extract the peak segments with OpenCV and assemble the recap.

OpenCV reads/writes frames only (no audio), so each clip's video is cut with
OpenCV and its audio is cut from the extracted WAV; ffmpeg then muxes the two
and concatenates all clips into the final recap.
"""
import os
import subprocess
import cv2


def extract_clip_opencv(video_path, start, end, out_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n_frames = int((end - start) * fps)
    written = 0
    while written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return written / fps  # actual seconds written


def mux_audio(silent_video, wav_path, start, duration, out_path):
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-i", silent_video, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", wav_path,
         "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-shortest", out_path],
        check=True,
    )


def build_recap(video_path, wav_path, peaks, clip_dir, recap_path):
    os.makedirs(clip_dir, exist_ok=True)
    clip_files = []
    for i, p in enumerate(peaks):
        silent = os.path.join(clip_dir, f"clip_{i:02d}_silent.mp4")
        final = os.path.join(clip_dir, f"clip_{i:02d}.mp4")
        secs = extract_clip_opencv(video_path, p["clip_start"], p["clip_end"], silent)
        if secs <= 0:
            raise RuntimeError(f"OpenCV read 0 frames from {video_path} – codec not supported?")
        mux_audio(silent, wav_path, p["clip_start"], secs, final)
        os.remove(silent)
        p["clip_file"] = final
        clip_files.append(final)

    list_file = os.path.join(clip_dir, "concat.txt")
    with open(list_file, "w") as f:
        for c in clip_files:
            f.write(f"file '{os.path.abspath(c)}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", list_file,
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", recap_path],
        check=True,
    )
    return recap_path


def duration_of(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path], capture_output=True, text=True, check=True)
    return float(out.stdout.strip())
