"""Step 2 – Split the input into its audio and video streams."""
import os
import subprocess
import cv2


def extract_audio(video_path: str, wav_path: str, sr: int = 16000) -> str:
    """Extract a mono 16 kHz WAV (the format Whisper expects)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le", wav_path],
        check=True,
    )
    return wav_path


def video_info(video_path: str) -> dict:
    """Read basic stream properties with OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "frames": frames,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration": frames / fps if fps else 0.0,
    }
    cap.release()
    return info


def video_codec(video_path: str) -> str:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "default=nw=1:nk=1", video_path],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def ensure_h264(video_path: str, out_dir: str = "data") -> str:
    """OpenCV's bundled FFmpeg cannot decode AV1/VP9 (YouTube's usual codecs).
    Re-encode once to H.264 so OpenCV can read every frame."""
    codec = video_codec(video_path)
    if codec == "h264":
        return video_path
    out = os.path.join(out_dir, "input_h264.mp4")
    if not os.path.exists(out):
        print(f"[media] Video codec is {codec}; converting to H.264 for OpenCV (one-time)...")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
             "-c:a", "aac", "-b:a", "128k", out], check=True)
    return out
