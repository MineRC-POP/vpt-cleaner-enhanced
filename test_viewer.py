import json
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


JSON_PATH = "test_predictions.json"
FRAMES_DIR = "frames"
MAX_IMG_W = 900
MAX_IMG_H = 700
POLLUTION_THRESHOLD = 0.5  # 污染判定阈值


class PredictionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("测试预测结果查看器")

        self.data = self.load_json(JSON_PATH)
        if not self.data:
            raise RuntimeError("test_predictions.json 为空或读取失败")

        self.index = 0
        self.tk_img = None

        main = tk.Frame(root)
        main.pack(fill="both", expand=True)

        # 左侧图片
        left = tk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        self.img_label = tk.Label(left, text="无图片", bg="black", fg="white")
        self.img_label.pack(fill="both", expand=True)

        # 右侧信息
        right = tk.Frame(main, width=420)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.info_text = tk.Text(right, wrap="word", font=("微软雅黑", 11))
        self.info_text.pack(fill="both", expand=True)

        # 底部按钮
        bottom = tk.Frame(root)
        bottom.pack(fill="x")

        self.prev_btn = tk.Button(bottom, text="上一个", command=self.prev_item)
        self.prev_btn.pack(side="left", padx=5, pady=5)

        self.next_btn = tk.Button(bottom, text="下一个", command=self.next_item)
        self.next_btn.pack(side="left", padx=5, pady=5)

        self.page_label = tk.Label(bottom, text="")
        self.page_label.pack(side="right", padx=10)

        # 快捷键
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())

        self.show_item()

    def load_json(self, path):
        if not os.path.exists(path):
            messagebox.showerror("错误", f"找不到文件: {path}")
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def resolve_image_path(self, image_field):
        norm = image_field.replace("\\", "/")

        if os.path.exists(norm):
            return norm

        base = os.path.basename(norm)
        p2 = os.path.join(FRAMES_DIR, base)
        if os.path.exists(p2):
            return p2

        if os.path.exists(image_field):
            return image_field

        return None

    def load_and_resize_image(self, path):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(MAX_IMG_W / w, MAX_IMG_H / h, 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)
        if scale != 1.0:
            img = img.resize((new_w, new_h), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def to_yes_no(self, value):
        try:
            return "是" if float(value) >= POLLUTION_THRESHOLD else "否"
        except Exception:
            return "否"

    def format_pollution_types(self, pollution_types):
        if not pollution_types:
            return "无"
        return "、".join(pollution_types)

    def format_info(self, item, real_path):
        lines = []
        lines.append(f"序号：{self.index + 1} / {len(self.data)}")
        lines.append("")
        lines.append(f"图片路径：{item.get('image', '')}")
        lines.append(f"实际路径：{real_path or '未找到'}")
        lines.append("")
        lines.append(f"预测游戏模式：{item.get('pred_game_mode', '')}")
        lines.append(f"预测界面类型：{item.get('pred_ui_type', '')}")
        lines.append(f"预测污染类型：{self.format_pollution_types(item.get('pred_pollution_types', []))}")
        lines.append("")
        lines.append("污染判断：")

        probs = item.get("pred_pollution_probs", {})
        ordered_keys = [
            "watermark_logo",
            "facecam_person",
            "text_overlay",
            "platform_ui_overlay",
            "border_frame",
            "other_artifact",
        ]

        cn_name = {
            "watermark_logo": "水印 / Logo",
            "facecam_person": "人像小窗",
            "text_overlay": "文字遮挡",
            "platform_ui_overlay": "平台界面叠层",
            "border_frame": "边框",
            "other_artifact": "其他杂质",
        }

        for key in ordered_keys:
            if key in probs:
                lines.append(f"  {cn_name.get(key, key)}：{self.to_yes_no(probs[key])}")

        # 如果 JSON 里有额外键，也一并显示
        for key, value in probs.items():
            if key not in ordered_keys:
                lines.append(f"  {cn_name.get(key, key)}：{self.to_yes_no(value)}")

        return "\n".join(lines)

    def show_item(self):
        item = self.data[self.index]
        img_path = self.resolve_image_path(item.get("image", ""))

        if img_path and os.path.exists(img_path):
            try:
                self.tk_img = self.load_and_resize_image(img_path)
                self.img_label.config(image=self.tk_img, text="")
            except Exception as e:
                self.img_label.config(image="", text=f"图片加载失败\n{e}")
                self.tk_img = None
        else:
            self.img_label.config(image="", text="图片不存在")
            self.tk_img = None

        info = self.format_info(item, img_path)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, info)

        self.page_label.config(text=f"{self.index + 1} / {len(self.data)}")

    def prev_item(self):
        if self.index > 0:
            self.index -= 1
            self.show_item()

    def next_item(self):
        if self.index < len(self.data) - 1:
            self.index += 1
            self.show_item()


def main():
    root = tk.Tk()
    root.geometry("1400x800")
    PredictionViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()