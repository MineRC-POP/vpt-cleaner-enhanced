import math
import subprocess
from pathlib import Path

ROOT = Path(r"G:\web_data_collector")
VIDEOS_DIR = ROOT / "videos"
FRAMES_DIR = ROOT / "frames"

TOTAL_TARGET_FRAMES = 1000
VIDEO_EXTS = {".mp4", ".webm"}

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# 是否清空输出目录里已有的 jpg
CLEAN_OUTPUT_DIR = False

# 可选：如果你的 ffmpeg 支持硬件解码，可以改成 True 试试
# 不是所有机器都稳定，先默认 False
USE_HWACCEL = False


def run_cmd(cmd):
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )


def find_videos(videos_dir: Path):
    if not videos_dir.exists():
        raise FileNotFoundError(f"视频目录不存在: {videos_dir}")
    files = [p for p in videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS]
    return sorted(files)


def get_duration_seconds(video_path: Path) -> float:
    cmd = [
        FFPROBE,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = run_cmd(cmd)
    text = result.stdout.strip()
    if not text:
        raise RuntimeError(f"ffprobe 未返回时长: {video_path}")
    duration = float(text)
    if duration <= 0:
        raise RuntimeError(f"视频时长无效: {video_path} -> {duration}")
    return duration


def allocate_counts(durations, total_target):
    """
    按时长占比分配，保证总和恰好等于 total_target。
    并尽量保证每个视频至少 1 帧（前提是 total_target >= 视频数）。
    """
    n = len(durations)
    if n == 0:
        return []

    total_duration = sum(durations)
    if total_duration <= 0:
        raise RuntimeError("所有视频总时长为 0")

    # 先给每个视频保底 1 帧
    base = [0] * n
    remain = total_target
    if total_target >= n:
        base = [1] * n
        remain = total_target - n

    raw = [d / total_duration * remain for d in durations]
    floors = [math.floor(x) for x in raw]
    alloc = [base[i] + floors[i] for i in range(n)]

    short = total_target - sum(alloc)
    remainders = [raw[i] - floors[i] for i in range(n)]
    order = sorted(range(n), key=lambda i: remainders[i], reverse=True)

    for i in order[:short]:
        alloc[i] += 1

    return alloc


def clean_output_dir(output_dir: Path):
    if not output_dir.exists():
        return
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in output_dir.glob(ext):
            p.unlink()


def extract_one_video(video_path: Path, duration: float, target_count: int, output_dir: Path):
    if target_count <= 0:
        return 0

    # 稍微往上抬一点，避免因为时间戳舍入导致少抽
    fps = (target_count + 0.25) / duration

    out_pattern = output_dir / f"{video_path.stem}_%06d.jpg"

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y"]

    if USE_HWACCEL:
        cmd += ["-hwaccel", "auto"]

    cmd += [
        "-threads", "0",
        "-i", str(video_path),
        "-an",
        "-sn",
        "-vf", f"fps={fps:.12f}",
        "-frames:v", str(target_count),
        "-q:v", "2",
        str(out_pattern),
    ]

    print(f"[extract] {video_path.name}")
    print(f"          duration={duration:.3f}s, target={target_count}, fps={fps:.8f}")

    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[error] ffmpeg 处理失败: {video_path.name}")
        print(e.stderr)
        return 0

    # 统计本次实际输出数量
    produced = len(list(output_dir.glob(f"{video_path.stem}_*.jpg")))
    return produced


def main():
    print(f"[info] videos: {VIDEOS_DIR}")
    print(f"[info] frames: {FRAMES_DIR}")
    print(f"[info] total target frames: {TOTAL_TARGET_FRAMES}")

    videos = find_videos(VIDEOS_DIR)
    if not videos:
        print("[error] 没找到 mp4/webm 视频")
        return

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAN_OUTPUT_DIR:
        print("[info] 正在清空输出目录中的旧图片...")
        clean_output_dir(FRAMES_DIR)

    infos = []
    for v in videos:
        try:
            dur = get_duration_seconds(v)
            infos.append((v, dur))
            print(f"[probe] {v.name}: {dur:.3f}s")
        except Exception as e:
            print(f"[warning] 跳过 {v.name}: {e}")

    if not infos:
        print("[error] 没有可处理的视频")
        return

    durations = [d for _, d in infos]
    counts = allocate_counts(durations, TOTAL_TARGET_FRAMES)

    print("\n[plan]")
    for (video, dur), cnt in zip(infos, counts):
        print(f"  {video.name}: {cnt} 帧 ({dur:.1f}s)")

    print(f"\n[info] 计划总帧数: {sum(counts)}")

    total_done = 0
    for (video, dur), cnt in zip(infos, counts):
        done = extract_one_video(video, dur, cnt, FRAMES_DIR)
        total_done += done
        print(f"[done] {video.name}: 实际输出 {done} 帧\n")

    print(f"[finished] 输出目录: {FRAMES_DIR}")
    print(f"[finished] 实际总输出: {total_done} 帧")


if __name__ == "__main__":
    main()