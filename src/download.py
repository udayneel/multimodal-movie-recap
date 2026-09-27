"""Step 1 – Get the input video (YouTube URL or local file)."""
import os
import shutil


def get_video(source: str, out_dir: str = "data") -> str:
    """Return a local path to the input video, downloading it if `source` is a URL."""
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(source):
        return source

    import yt_dlp  # imported lazily so local-file runs don't need it

    out_path = os.path.join(out_dir, "input.mp4")
    if os.path.exists(out_path):
        print(f"[download] Using cached {out_path}")
        return out_path

    opts = {
        "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480]/best",
        "merge_output_format": "mp4",
        "outtmpl": out_path,
        "quiet": False,
    }
    # YouTube often blocks cloud IPs (Colab) with "Sign in to confirm you're not a bot".
    # Exporting your browser's YouTube cookies to data/cookies.txt fixes this.
    cookies = os.path.join(out_dir, "cookies.txt")
    if os.path.exists(cookies):
        opts["cookiefile"] = cookies
        print("[download] Using cookies from", cookies)
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([source])
    print(f"[download] Saved to {out_path}")
    return out_path


def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it (apt install ffmpeg).")
