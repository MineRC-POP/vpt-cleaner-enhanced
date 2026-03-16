import os
import re
import json
import time
import random
import pathlib
from typing import Dict, List, Optional, Iterable

import requests
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


# =========================
# 配置区
# =========================

SEARCH_KEYWORDS = [
    "minecraft survival longplay",
    "minecraft gameplay no webcam",
    "minecraft gameplay survival mode",
    "minecraft survival tutorial",
    "minecraft survival guide",
    "minecraft survival let's play",
    "minecraft survival for beginners",
    "minecraft beginners guide",
    "ultimate minecraft starter guide",
    "minecraft survival guide 1.16",
    "minecraft how to start a new survival world",
    "minecraft survival fresh start",
    "minecraft survival let's play episode 1",
    "let's play minecraft episode 1",
    "minecraft survival 101",
    "minecraft survival learning to play",
    "how to play minecraft survival",
    "how to play minecraft",
    "minecraft survival basic",
    "minecraft survival for noobs",
    "minecraft survival for dummies",
    "how to play minecraft for beginners",
    "minecraft survival tutorial series",
    "minecraft survival new world",
    "minecraft survival a new beginning",
    "minecraft survival episodio 1",
    "minecraft survival 1",
    "minecraft survival 1. bölüm",
    "i made a new minecraft survival world",
]

BLACKLIST_TERMS = [
    "ps3",
    "ps4",
    "ps5",
    "xbox 360",
    "playstation",
    "timelapse",
    "multiplayer",
    "minecraft pe",
    "pocket edition",
    "skyblock",
    "realistic minecraft",
    "how to install",
    "how to download",
    "realmcraft",
    "animation",
]

OUTPUT_DIR = pathlib.Path("yt_dataset_out")
META_DIR = OUTPUT_DIR / "meta"
TRANSCRIPT_DIR = OUTPUT_DIR / "transcripts"
VIDEO_DIR = OUTPUT_DIR / "videos"

# 模式:
# "official" = YouTube Data API
# "unofficial" = 直接走 yt-dlp 的 ytsearch
SEARCH_MODE = "unofficial"

# 如果是 official 模式，需要填 API Key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

MAX_RESULTS_PER_QUERY = 30
SLEEP_BETWEEN_QUERIES = (1.2, 2.5)

DOWNLOAD_VIDEO = False
DOWNLOAD_SUBTITLES = True
EXTRACT_INFO_ONLY = True   # True = 只抓元数据，不下载视频

# 时间过滤，可选，例如 "2020-01-01T00:00:00Z"
PUBLISHED_AFTER = None

# 只保留标题/描述语言大概率是英文的视频（简单启发式）
PREFER_ENGLISH = False


# =========================
# 工具函数
# =========================

def ensure_dirs():
    for d in [OUTPUT_DIR, META_DIR, TRANSCRIPT_DIR, VIDEO_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def contains_blacklist(text: str, blacklist: List[str]) -> bool:
    low = text.lower()
    return any(term.lower() in low for term in blacklist)

def looks_english(text: str) -> bool:
    if not text:
        return True
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return ascii_ratio > 0.9

def save_jsonl(path: pathlib.Path, rows: Iterable[Dict]):
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# 搜索：官方 API
# =========================

def official_search_youtube(query: str, max_results: int = 20) -> List[Dict]:
    """
    使用 YouTube Data API v3 的 search.list 搜索视频
    """
    if not YOUTUBE_API_KEY:
        raise RuntimeError("官方模式需要设置环境变量 YOUTUBE_API_KEY")

    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "key": YOUTUBE_API_KEY,
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": min(max_results, 50),
        "safeSearch": "none",
    }
    if PUBLISHED_AFTER:
        params["publishedAfter"] = PUBLISHED_AFTER

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    results = []
    for item in data.get("items", []):
        vid = item["id"]["videoId"]
        snippet = item.get("snippet", {})
        results.append({
            "video_id": vid,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channel_title": snippet.get("channelTitle"),
            "published_at": snippet.get("publishedAt"),
            "source_query": query,
            "url": f"https://www.youtube.com/watch?v={vid}",
            "search_backend": "official_api",
        })
    return results


# =========================
# 搜索：非官方（yt-dlp ytsearch）
# =========================

def unofficial_search_youtube(query: str, max_results: int = 20) -> List[Dict]:
    """
    用 yt-dlp 的 ytsearch 直接搜，不依赖官方 API
    """
    search_expr = f"ytsearch{max_results}:{query}"

    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,   # 先拿平面结果，快一点
        "nocheckcertificate": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_expr, download=False)

    results = []
    for entry in info.get("entries", []) or []:
        if not entry:
            continue
        vid = entry.get("id")
        url = entry.get("url") or f"https://www.youtube.com/watch?v={vid}"
        results.append({
            "video_id": vid,
            "title": entry.get("title"),
            "description": entry.get("description"),
            "channel_title": entry.get("channel"),
            "published_at": entry.get("upload_date"),
            "source_query": query,
            "url": url if str(url).startswith("http") else f"https://www.youtube.com/watch?v={vid}",
            "search_backend": "yt_dlp_search",
        })
    return results


# =========================
# 详情提取
# =========================

import yt_dlp

COOKIE_FILE = r"E:\Users\Evidence\Documents\minerc\web_data_collector\cookies.txt"

def extract_video_info(url: str) -> dict:
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "nocheckcertificate": True,
        "verbose": True,

        # 改这里：直接从 cookies.txt 读
        "cookiefile": COOKIE_FILE,

        "extractor_args": {
            "youtube": {
                "player_client": ["default"]
            }
        },

        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/145.0.0.0 Safari/537.36"
            )
        },

        # 如果你已经装好了 deno，可以加上
        # "js_runtimes": {
        #     "deno": {
        #         "path": r"C:\Users\Evidence\.deno\bin\deno.exe"
        #     }
        # },

        # 可选：需要时再开
        # "remote_components": {"ejs:github"},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "id": info.get("id"),
        "title": info.get("title"),
        "description": info.get("description"),
        "channel": info.get("channel"),
        "channel_id": info.get("channel_id"),
        "uploader": info.get("uploader"),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "upload_date": info.get("upload_date"),
        "tags": info.get("tags"),
        "categories": info.get("categories"),
        "language": info.get("language"),
        "subtitles": list((info.get("subtitles") or {}).keys()),
        "automatic_captions": list((info.get("automatic_captions") or {}).keys()),
        "webpage_url": info.get("webpage_url"),
        "thumbnail": info.get("thumbnail"),
    }


# =========================
# 字幕提取
# =========================

def fetch_transcript(video_id: str) -> Optional[List[Dict]]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        return transcript
    except Exception:
        return None


# =========================
# 视频下载（可选）
# =========================

def download_video(url: str, out_dir: pathlib.Path):
    ydl_opts = {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "nocheckcertificate": True,
        # 你也可以限制格式，例如 720p:
        # "format": "bv*[height<=720]+ba/b[height<=720]/b"
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


# =========================
# 主流程
# =========================

def search_videos(query: str, max_results: int) -> List[Dict]:
    if SEARCH_MODE == "official":
        return official_search_youtube(query, max_results=max_results)
    elif SEARCH_MODE == "unofficial":
        return unofficial_search_youtube(query, max_results=max_results)
    else:
        raise ValueError(f"未知 SEARCH_MODE: {SEARCH_MODE}")

def filter_candidate(item: Dict) -> bool:
    title = normalize_text(item.get("title"))
    desc = normalize_text(item.get("description"))
    merged = f"{title}\n{desc}"

    if not title:
        return False
    if contains_blacklist(merged, BLACKLIST_TERMS):
        return False
    if PREFER_ENGLISH and not looks_english(merged):
        return False
    return True

def main():
    ensure_dirs()

    seen_ids = set()
    meta_jsonl = META_DIR / "videos.jsonl"

    for query in SEARCH_KEYWORDS:
        print(f"[search] {query}")
        try:
            results = search_videos(query, MAX_RESULTS_PER_QUERY)
        except Exception as e:
            print(f"[search][error] {query}: {e}")
            continue

        print(f"  found={len(results)}")

        for item in results:
            vid = item.get("video_id")
            if not vid or vid in seen_ids:
                continue
            if not filter_candidate(item):
                continue

            url = item["url"]

            # 拉详细信息
            try:
                detailed = extract_video_info(url)
            except Exception as e:
                print(f"[info][skip] {url}: {e}")
                continue

            # 二次过滤
            merged = normalize_text(detailed.get("title")) + "\n" + normalize_text(detailed.get("description"))
            if contains_blacklist(merged, BLACKLIST_TERMS):
                continue
            if PREFER_ENGLISH and not looks_english(merged):
                continue

            row = {
                **item,
                **detailed,
            }

            save_jsonl(meta_jsonl, [row])
            seen_ids.add(vid)
            print(f"  [saved] {vid} | {row.get('title')}")

            # 字幕
            if DOWNLOAD_SUBTITLES:
                transcript = fetch_transcript(vid)
                if transcript:
                    with (TRANSCRIPT_DIR / f"{vid}.json").open("w", encoding="utf-8") as f:
                        json.dump(transcript, f, ensure_ascii=False, indent=2)

            # 视频
            if DOWNLOAD_VIDEO and not EXTRACT_INFO_ONLY:
                try:
                    download_video(url, VIDEO_DIR)
                except Exception as e:
                    print(f"[download][skip] {url}: {e}")

        time.sleep(random.uniform(*SLEEP_BETWEEN_QUERIES))

    print("done.")


if __name__ == "__main__":
    main()