import json
import pathlib
from typing import Dict, Iterator

import yt_dlp

JSONL_FILE = pathlib.Path("yt_dataset_out/meta/videos.jsonl")
COOKIE_FILE = pathlib.Path("cookies.txt")
VIDEOS_DIR = pathlib.Path("videos")

# 720p 无音频；优先精确 720p，没有就退到 <=720p
DOWNLOAD_FORMAT = "bestvideo[height=720][acodec=none]/bestvideo[height<=720][acodec=none]"


def iter_jsonl(path: pathlib.Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[jsonl][skip] line {line_no}: {e}")


def make_ydl_opts() -> Dict:
    opts = {
        "format": DOWNLOAD_FORMAT,
        "outtmpl": str(VIDEOS_DIR / "%(id)s_720p_noaudio.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            )
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["default"]
            }
        },
        "js_runtimes": {"deno": {}},
        # 如果后续需要远程组件，可以取消下一行注释
        # "remote_components": {"ejs:github"},
    }

    if COOKIE_FILE.exists():
        opts["cookiefile"] = str(COOKIE_FILE)
    else:
        print("[warn] cookies.txt not found; some YouTube videos may fail")

    return opts


def already_downloaded(video_id: str) -> bool:
    patterns = [
        f"{video_id}_720p_noaudio.*",
    ]
    for pattern in patterns:
        if any(VIDEOS_DIR.glob(pattern)):
            return True
    return False


def main() -> None:
    if not JSONL_FILE.exists():
        raise FileNotFoundError(f"JSONL file not found: {JSONL_FILE.resolve()}")

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    rows = list(iter_jsonl(JSONL_FILE))
    print(f"[load] rows={len(rows)} from {JSONL_FILE}")

    ydl_opts = make_ydl_opts()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, row in enumerate(rows, 1):
            video_id = row.get("id") or row.get("video_id")
            url = row.get("webpage_url") or row.get("url")
            title = row.get("title") or "<untitled>"

            if not video_id or not url:
                print(f"[skip] #{i}: missing id/url")
                continue

            if already_downloaded(video_id):
                print(f"[skip] #{i}: already downloaded {video_id} | {title}")
                continue

            print(f"[download] #{i}: {video_id} | {title}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"[error] #{i}: {video_id} | {e}")


if __name__ == "__main__":
    main()
