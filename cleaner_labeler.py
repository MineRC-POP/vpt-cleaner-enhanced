import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
# ===== AI 自动标注 =====
from tqdm import tqdm
from val import load_model, predict_one, choose_device
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QShortcut,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

GAME_MODES = [
    ("survival", "生存模式 [1]"),
    ("non_survival", "非生存模式 [2]"),
    ("not_minecraft", "不是 Minecraft [3]"),
    ("unknown_mode", "无法判断 [4]"),
]

POLLUTION_TYPES = [
    ("watermark_logo", "水印 / Logo [Q]"),
    ("facecam_person", "人像 / Facecam [W]"),
    ("text_overlay", "字幕 / 文字 [E]"),
    ("platform_ui_overlay", "平台浮层 / 播放器 [R]"),
    ("border_frame", "边框 / 模板 [T]"),
    ("other_artifact", "其他污染 [Y]"),
]

UI_TYPES = [
    ("none", "未打开额外 UI [Z]"),
    ("chat", "聊天 [X]"),
    ("pause_menu", "暂停菜单 [C]"),
    ("settings", "设置 / 选项 [V]"),
    ("inventory", "背包 [B]"),
    ("container", "容器 [N]"),
    ("death_screen", "死亡界面 [M]"),
    ("loading", "加载界面 [,]"),
    ("other_ui", "其他 UI [.]"),
]


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None
        self.setText("未加载图片")
        self.setWordWrap(True)

    def set_original_pixmap(self, pixmap: QPixmap | None):
        self._pixmap = pixmap
        self._update_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self):
        if not self._pixmap or self._pixmap.isNull():
            self.setText("图片加载失败")
            self.setPixmap(QPixmap())
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setPixmap(scaled)


class AnnotatorWindow(QMainWindow):
    def __init__(self, image_dir: Path, save_path: Path):
        super().__init__()
        self.image_dir = image_dir
        self.save_path = save_path
        self.images = self._find_images(image_dir)
        self.annotations = {}
        self.current_index = 0
        self._loading_widgets = False
        self._current_pixmap = None

        self.setWindowTitle("Minecraft 帧标注器")
        self.resize(1400, 900)
        self._build_ui()
        self._apply_theme()
        self._load_existing_annotations()

        if not self.images:
            QMessageBox.warning(self, "没有图片", f"在目录中没有找到图片：\n{image_dir}")
        self.load_current_image()
        self._bind_shortcuts()
        # ===== AI =====
        self.ai_model = None
        self.ai_device = None
        self._init_ai_model()
        
    def ai_annotate_next_100(self):

        if self.ai_model is None:
            QMessageBox.warning(self, "AI未加载", "模型 runs/best.pt 未加载")
            return

        start = self.current_index
        end = min(len(self.images), start + 100)

        paths = self.images[start:end]

        print(f"[AI] 标注 {len(paths)} 张图片")

        for path in tqdm(paths, desc="AI标注"):

            result = predict_one(
                self.ai_model,
                path,
                self.ai_device,
                pollution_threshold=0.5,
            )

            try:
                key = str(path.relative_to(self.image_dir.parent).as_posix())
            except ValueError:
                key = str(path.name)

            ann = {
                "game_mode": result["game_mode"],
                "pollution_types": result["pollution_types"],
                "ui_type": result["ui_type"],
                "uncertain": False,
                "pollution_status": result["pollution_status"],
                "ui_state": result["ui_state"],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

            self.annotations[key] = ann

        self.save_annotations()

        QMessageBox.information(
            self,
            "AI标注完成",
            f"完成 {len(paths)} 张 AI 标注"
        )

        self.load_current_image()
    def _find_images(self, image_dir: Path):
        if not image_dir.exists():
            return []
        files = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        return sorted(files, key=lambda p: natural_key(p.name))
    def _init_ai_model(self):
        try:
            ckpt = Path("runs/best.pt")
            if not ckpt.exists():
                self.statusBar().showMessage("AI模型未找到 runs/best.pt")
                return

            self.ai_device = choose_device()
            print(f"[AI] device = {self.ai_device}")

            self.ai_model = load_model(ckpt, self.ai_device)

            self.statusBar().showMessage("AI模型加载完成")

        except Exception as e:
            QMessageBox.warning(self, "AI模型加载失败", str(e))
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # Left: image
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.image_title = QLabel("未加载")
        self.image_title.setObjectName("ImageTitle")
        left_layout.addWidget(self.image_title)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        self.image_label = ImageLabel()
        scroll_area.setWidget(self.image_label)
        left_layout.addWidget(scroll_area)

        self.bottom_info = QLabel("")
        self.bottom_info.setObjectName("BottomInfo")
        left_layout.addWidget(self.bottom_info)

        splitter.addWidget(left_panel)

        # Right: controls
        right_panel = QWidget()
        right_panel.setMinimumWidth(390)
        right_panel.setMaximumWidth(520)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)

        header_card = QFrame()
        header_card.setObjectName("HeaderCard")
        header_layout = QVBoxLayout(header_card)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(8)

        self.progress_label = QLabel("0 / 0")
        self.progress_label.setObjectName("ProgressLabel")
        header_layout.addWidget(self.progress_label)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("← 上一个 [A]")
        self.prev_btn.clicked.connect(self.go_prev)
        self.next_btn = QPushButton("下一个 [D/Enter] →")
        self.next_btn.clicked.connect(self.go_next)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        header_layout.addLayout(nav_row)

        save_row = QHBoxLayout()
        self.save_btn = QPushButton("保存 [Ctrl+S]")
        self.save_btn.clicked.connect(self.save_annotations)
        self.clear_btn = QPushButton("清空当前 [Backspace]")
        self.clear_btn.clicked.connect(self.clear_current_annotation)
        self.ai_btn = QPushButton("AI标注后100张")
        self.ai_btn.clicked.connect(self.ai_annotate_next_100)
        save_row.addWidget(self.ai_btn)
        save_row.addWidget(self.save_btn)
        save_row.addWidget(self.clear_btn)
        header_layout.addLayout(save_row)

        right_layout.addWidget(header_card)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)

        right_content = QWidget()
        right_content_layout = QVBoxLayout(right_content)
        right_content_layout.setContentsMargins(0, 0, 0, 0)
        right_content_layout.setSpacing(8)
        right_content_layout.addWidget(self._build_game_mode_box())
        right_content_layout.addWidget(self._build_pollution_box())
        right_content_layout.addWidget(self._build_ui_box())
        right_content_layout.addWidget(self._build_meta_box())
        right_content_layout.addWidget(self._build_help_box())
        right_content_layout.addStretch(1)

        right_scroll.setWidget(right_content)
        right_layout.addWidget(right_scroll, 1)

        splitter.addWidget(right_panel)
        splitter.setSizes([980, 420])

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    def _build_game_mode_box(self):
        box = QGroupBox("1. 游戏模式")
        layout = QVBoxLayout(box)
        self.game_mode_group = QButtonGroup(self)
        self.game_mode_group.setExclusive(True)
        self.game_mode_buttons = {}
        for idx, (value, label) in enumerate(GAME_MODES):
            btn = QRadioButton(label)
            self.game_mode_group.addButton(btn, idx)
            self.game_mode_buttons[value] = btn
            btn.toggled.connect(self.on_field_changed)
            layout.addWidget(btn)
        return box

    def _build_pollution_box(self):
        box = QGroupBox("2. 污染类型（多选）")
        layout = QVBoxLayout(box)
        self.pollution_checks = {}
        for value, label in POLLUTION_TYPES:
            cb = QCheckBox(label)
            cb.toggled.connect(self.on_field_changed)
            self.pollution_checks[value] = cb
            layout.addWidget(cb)

        self.pollution_status_label = QLabel("自动推导：无污染")
        self.pollution_status_label.setObjectName("DerivedLabel")
        layout.addWidget(self.pollution_status_label)
        return box

    def _build_ui_box(self):
        box = QGroupBox("3. 当前 UI 类型")
        layout = QVBoxLayout(box)
        self.ui_type_group = QButtonGroup(self)
        self.ui_type_group.setExclusive(True)
        self.ui_type_buttons = {}
        for idx, (value, label) in enumerate(UI_TYPES):
            btn = QRadioButton(label)
            self.ui_type_group.addButton(btn, idx)
            self.ui_type_buttons[value] = btn
            btn.toggled.connect(self.on_field_changed)
            layout.addWidget(btn)

        self.ui_state_label = QLabel("自动推导：UI 未打开")
        self.ui_state_label.setObjectName("DerivedLabel")
        layout.addWidget(self.ui_state_label)
        return box

    def _build_meta_box(self):
        box = QGroupBox("4. 其他")
        layout = QGridLayout(box)

        self.uncertain_checkbox = QCheckBox("无法判断 [U]")
        self.uncertain_checkbox.toggled.connect(self.on_field_changed)
        layout.addWidget(self.uncertain_checkbox, 0, 0, 1, 2)

        self.completion_label = QLabel("当前状态：未标注")
        self.completion_label.setObjectName("DerivedLabel")
        layout.addWidget(self.completion_label, 1, 0, 1, 2)

        self.save_path_label = QLabel(f"保存到：{self.save_path}")
        self.save_path_label.setWordWrap(True)
        layout.addWidget(self.save_path_label, 2, 0, 1, 2)
        return box

    def _build_help_box(self):
        box = QGroupBox("5. 快捷键")
        layout = QVBoxLayout(box)
        help_text = (
            "A 上一个\n"
            "D / Enter 下一个\n"
            "Ctrl+S 保存\n"
            "Backspace 清空当前\n"
            "1~4 游戏模式\n"
            "Q/W/E/R/T/Y 污染类型开关\n"
            "Z/X/C/V/B/N/M/,/. 选择 UI 类型\n"
            "U 切换无法判断"
        )
        label = QLabel(help_text)
        label.setWordWrap(True)
        layout.addWidget(label)
        return box

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #181818;
                color: #f5f5f5;
                font-size: 14px;
            }
            QMainWindow {
                background: #181818;
            }
            QGroupBox {
                border: 1px solid #ff8c2a;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 12px;
                font-weight: 600;
                background: #222222;
            }
            QFrame#HeaderCard {
                background: #222222;
                border: 1px solid #ff8c2a;
                border-radius: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #ffb366;
            }
            QPushButton {
                background: #ff8c2a;
                color: #111111;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #ff9d49;
            }
            QPushButton:pressed {
                background: #e67810;
            }
            QRadioButton, QCheckBox {
                spacing: 8px;
                padding: 3px 0;
            }
            QRadioButton::indicator, QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator:unchecked, QCheckBox::indicator:unchecked {
                border: 1px solid #ffb366;
                background: #ffffff;
                border-radius: 4px;
            }
            QRadioButton::indicator:checked, QCheckBox::indicator:checked {
                border: 1px solid #ff8c2a;
                background: #ff8c2a;
                border-radius: 4px;
            }
            QScrollArea {
                border: 1px solid #333333;
                border-radius: 10px;
                background: #0f0f0f;
            }
            QScrollBar:vertical {
                background: #111111;
                width: 12px;
                margin: 2px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #ff8c2a;
                min-height: 24px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QLabel#ImageTitle {
                font-size: 18px;
                font-weight: 700;
                color: #ffb366;
            }
            QLabel#BottomInfo, QLabel#ProgressLabel, QLabel#DerivedLabel {
                color: #ffd4a3;
            }
            QStatusBar {
                background: #101010;
                color: #ffffff;
            }
            """
        )

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("A"), self, activated=self.go_prev)
        QShortcut(QKeySequence("D"), self, activated=self.go_next)
        QShortcut(QKeySequence(Qt.Key_Return), self, activated=self.go_next)
        QShortcut(QKeySequence(Qt.Key_Enter), self, activated=self.go_next)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_annotations)
        QShortcut(QKeySequence(Qt.Key_Backspace), self, activated=self.clear_current_annotation)
        QShortcut(QKeySequence("U"), self, activated=self.toggle_uncertain)

        for i, (value, _) in enumerate(GAME_MODES, start=1):
            QShortcut(QKeySequence(str(i)), self, activated=lambda v=value: self.select_game_mode(v))

        pollution_keys = ["Q", "W", "E", "R", "T", "Y"]
        for key, (value, _) in zip(pollution_keys, POLLUTION_TYPES):
            QShortcut(QKeySequence(key), self, activated=lambda v=value: self.toggle_pollution(v))

        ui_keys = ["Z", "X", "C", "V", "B", "N", "M", ",", "."]
        for key, (value, _) in zip(ui_keys, UI_TYPES):
            QShortcut(QKeySequence(key), self, activated=lambda v=value: self.select_ui_type(v))

    def _load_existing_annotations(self):
        if not self.save_path.exists():
            return
        try:
            with self.save_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.annotations = data.get("items", {})
            self.current_index = int(data.get("__meta__", {}).get("current_index", 0))
            if self.current_index < 0 or self.current_index >= max(len(self.images), 1):
                self.current_index = 0
            self.statusBar().showMessage(f"已加载已有标注：{self.save_path}")
        except Exception as e:
            QMessageBox.warning(self, "读取失败", f"读取已有标注失败：\n{e}")
            self.annotations = {}
            self.current_index = 0

    def current_image_path(self) -> Path | None:
        if not self.images:
            return None
        return self.images[self.current_index]

    def current_key(self) -> str | None:
        path = self.current_image_path()
        if path is None:
            return None
        try:
            return str(path.relative_to(self.image_dir.parent).as_posix())
        except ValueError:
            return str(path.name)

    def default_annotation(self):
        return {
            "game_mode": None,
            "pollution_types": [],
            "ui_type": "none",
            "uncertain": False,
            "pollution_status": "clean",
            "ui_state": "ui_closed",
            "updated_at": None,
        }

    def get_current_annotation(self):
        key = self.current_key()
        if key is None:
            return self.default_annotation()
        ann = self.annotations.get(key)
        if ann is None:
            return self.default_annotation()
        merged = self.default_annotation()
        merged.update(ann)
        merged["pollution_types"] = list(ann.get("pollution_types", []))
        return merged

    def collect_form_data(self):
        game_mode = None
        for value, btn in self.game_mode_buttons.items():
            if btn.isChecked():
                game_mode = value
                break
        pollution_types = [value for value, cb in self.pollution_checks.items() if cb.isChecked()]
        ui_type = "none"
        for value, btn in self.ui_type_buttons.items():
            if btn.isChecked():
                ui_type = value
                break
        uncertain = self.uncertain_checkbox.isChecked()
        pollution_status = "polluted" if pollution_types else "clean"
        ui_state = "ui_open" if ui_type != "none" else "ui_closed"
        return {
            "game_mode": game_mode,
            "pollution_types": pollution_types,
            "ui_type": ui_type,
            "uncertain": uncertain,
            "pollution_status": pollution_status,
            "ui_state": ui_state,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def apply_annotation_to_form(self, annotation: dict):
        self._loading_widgets = True
        try:
            for btn in self.game_mode_buttons.values():
                btn.setAutoExclusive(False)
                btn.setChecked(False)
                btn.setAutoExclusive(True)
            if annotation.get("game_mode") in self.game_mode_buttons:
                self.game_mode_buttons[annotation["game_mode"]].setChecked(True)

            selected_pollution = set(annotation.get("pollution_types", []))
            for value, cb in self.pollution_checks.items():
                cb.setChecked(value in selected_pollution)

            for btn in self.ui_type_buttons.values():
                btn.setAutoExclusive(False)
                btn.setChecked(False)
                btn.setAutoExclusive(True)
            ui_value = annotation.get("ui_type") or "none"
            if ui_value not in self.ui_type_buttons:
                ui_value = "none"
            self.ui_type_buttons[ui_value].setChecked(True)

            self.uncertain_checkbox.setChecked(bool(annotation.get("uncertain", False)))
            self.refresh_derived_labels(annotation)
        finally:
            self._loading_widgets = False

    def refresh_derived_labels(self, annotation=None):
        if annotation is None:
            annotation = self.collect_form_data()
        if annotation["pollution_status"] == "polluted":
            self.pollution_status_label.setText("自动推导：有污染")
        else:
            self.pollution_status_label.setText("自动推导：无污染")

        if annotation["ui_state"] == "ui_open":
            self.ui_state_label.setText(f"自动推导：UI 已打开（{annotation['ui_type']}）")
        else:
            self.ui_state_label.setText("自动推导：UI 未打开")

        marked = self.is_annotation_meaningful(annotation)
        self.completion_label.setText("当前状态：已标注" if marked else "当前状态：未标注")

    def is_annotation_meaningful(self, annotation: dict):
        return bool(
            annotation.get("game_mode")
            or annotation.get("pollution_types")
            or annotation.get("ui_type") not in (None, "none")
            or annotation.get("uncertain")
        )

    def on_field_changed(self):
        if self._loading_widgets:
            return
        ann = self.collect_form_data()
        self.refresh_derived_labels(ann)
        self.save_current_annotation(ann)

    def save_current_annotation(self, annotation=None):
        key = self.current_key()
        if key is None:
            return
        if annotation is None:
            annotation = self.collect_form_data()
        if self.is_annotation_meaningful(annotation):
            self.annotations[key] = annotation
        else:
            self.annotations.pop(key, None)
        self.save_annotations(show_status_only=True)

    def save_annotations(self, show_status_only=False):
        data = {
            "__meta__": {
                "version": 1,
                "image_dir": str(self.image_dir),
                "current_index": self.current_index,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "image_count": len(self.images),
                "annotated_count": len(self.annotations),
            },
            "items": self.annotations,
        }
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.save_path.with_suffix(self.save_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.save_path)
        msg = f"已保存：{self.save_path.name}（{len(self.annotations)}/{len(self.images)}）"
        self.statusBar().showMessage(msg)
        if not show_status_only:
            self.refresh_side_info()

    def load_current_image(self):
        if not self.images:
            self.image_title.setText("frames 文件夹中没有可用图片")
            self.image_label.setText("请把图片放进 frames 文件夹")
            self.progress_label.setText("0 / 0")
            self.bottom_info.setText("")
            self.refresh_derived_labels(self.default_annotation())
            return

        path = self.current_image_path()
        self.image_title.setText(path.name)
        self.progress_label.setText(f"{self.current_index + 1} / {len(self.images)}")
        self.bottom_info.setText(str(path))
        pixmap = QPixmap(str(path))
        self.image_label.set_original_pixmap(pixmap)

        ann = self.get_current_annotation()
        self.apply_annotation_to_form(ann)
        self.refresh_side_info()

    def refresh_side_info(self):
        annotated = len(self.annotations)
        total = len(self.images)
        current = self.current_index + 1 if total else 0
        self.progress_label.setText(f"{current} / {total}    已标注：{annotated}")

    def go_prev(self):
        if not self.images:
            return
        self.save_current_annotation()
        self.current_index = max(0, self.current_index - 1)
        self.load_current_image()

    def go_next(self):
        if not self.images:
            return
        self.save_current_annotation()
        self.current_index = min(len(self.images) - 1, self.current_index + 1)
        self.load_current_image()

    def clear_current_annotation(self):
        if not self.images:
            return
        self.apply_annotation_to_form(self.default_annotation())
        key = self.current_key()
        if key is not None:
            self.annotations.pop(key, None)
        self.save_annotations(show_status_only=True)
        self.refresh_side_info()
        self.statusBar().showMessage("已清空当前图片标注")

    def toggle_uncertain(self):
        self.uncertain_checkbox.setChecked(not self.uncertain_checkbox.isChecked())

    def select_game_mode(self, value: str):
        btn = self.game_mode_buttons.get(value)
        if btn:
            btn.setChecked(True)

    def toggle_pollution(self, value: str):
        cb = self.pollution_checks.get(value)
        if cb:
            cb.setChecked(not cb.isChecked())

    def select_ui_type(self, value: str):
        btn = self.ui_type_buttons.get(value)
        if btn:
            btn.setChecked(True)

    def closeEvent(self, event):
        try:
            self.save_current_annotation()
            self.save_annotations(show_status_only=True)
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"退出前保存失败：\n{e}")
        super().closeEvent(event)


def main():
    base_dir = Path.cwd()
    image_dir = base_dir / "frames"
    save_path = base_dir / "annotations.json"

    if len(sys.argv) >= 2:
        image_dir = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        save_path = Path(sys.argv[2])

    app = QApplication(sys.argv)
    app.setApplicationName("Minecraft 帧标注器")
    win = AnnotatorWindow(image_dir=image_dir, save_path=save_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
