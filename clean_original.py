import json
import shutil
from pathlib import Path


def safe_copy_with_rename(src: Path, dst_dir: Path) -> Path:
    """
    复制文件到目标目录。
    如果文件名已存在，则自动在后面加 _1, _2 ...
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if not dst.exists():
        shutil.copy2(src, dst)
        return dst

    stem = src.stem
    suffix = src.suffix
    i = 1
    while True:
        new_dst = dst_dir / f"{stem}_{i}{suffix}"
        if not new_dst.exists():
            shutil.copy2(src, new_dst)
            return new_dst
        i += 1


def main():
    # ===== 可按需修改 =====
    annotations_path = Path("annotations.json")
    output_dir = Path("frame_true")
    # ====================

    if not annotations_path.exists():
        print(f"[错误] 找不到标注文件: {annotations_path}")
        return

    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", {})
    meta = data.get("__meta__", {})

    # annotations.json 所在目录
    json_dir = annotations_path.parent.resolve()

    # 从 __meta__.image_dir 取图片目录（如果有）
    meta_image_dir = meta.get("image_dir")
    if meta_image_dir:
        meta_image_dir = Path(meta_image_dir)
    else:
        meta_image_dir = None

    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    copied = 0
    missing = 0
    failed = 0

    for rel_path_str in items.keys():
        total += 1

        # 统一路径分隔符
        norm_rel = rel_path_str.replace("\\", "/")
        rel_path = Path(norm_rel)

        candidate_paths = []

        # 1. 相对于 annotations.json 所在目录
        candidate_paths.append((json_dir / rel_path).resolve())

        # 2. 如果 meta 里有 image_dir，则尝试用它拼接
        #    比如 rel_path = frames/xxx.jpg，meta_image_dir = G:\web_data_collector\frames
        #    那么优先尝试 image_dir / 文件名
        if meta_image_dir is not None:
            candidate_paths.append((meta_image_dir / rel_path.name).resolve())
            candidate_paths.append((meta_image_dir.parent / rel_path).resolve())
            candidate_paths.append(meta_image_dir.resolve() / rel_path)

        # 去重，按顺序保留
        unique_candidates = []
        seen = set()
        for p in candidate_paths:
            ps = str(p)
            if ps not in seen:
                seen.add(ps)
                unique_candidates.append(p)

        src_path = None
        for p in unique_candidates:
            if p.exists() and p.is_file():
                src_path = p
                break

        if src_path is None:
            missing += 1
            print(f"[缺失] {rel_path_str}")
            continue

        try:
            dst_path = safe_copy_with_rename(src_path, output_dir)
            copied += 1
            print(f"[复制] {src_path} -> {dst_path}")
        except Exception as e:
            failed += 1
            print(f"[失败] {src_path} -> {e}")

    print("\n====== 完成 ======")
    print(f"总条目数: {total}")
    print(f"成功复制: {copied}")
    print(f"缺失文件: {missing}")
    print(f"复制失败: {failed}")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()