import os
from PyQt6.QtCore import Qt, QUrl, QPoint, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt6.QtMultimedia import QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTextEdit, QSizePolicy, QGraphicsDropShadowEffect, QFrame
)
from pathlib import Path
from stream_worker import InferenceThread

import cv2, numpy as np

from titlebar import TitleBar

import model


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suspicious Behavior Detector")
        self.setMinimumSize(960, 640)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self.file_path = None

        # shadow wrapper
        shadow_wrapper = QVBoxLayout(self)
        shadow_wrapper.setContentsMargins(20, 20, 20, 20)

        self.container = QFrame()
        self.container.setObjectName("Container")
        self.container.setStyleSheet("""
            #Container {
                background-color: #121212;
                border-radius: 10px;
            }
        """)
        self.container.setGraphicsEffect(self._create_shadow())
        self.container.setLayout(self._build_ui())

        self.container.resizeEvent = self.container_resize_event
        self.resize_grip = ResizeGrip(self.container)
        self.resize_grip.raise_()
        self.update_resize_grip_pos()

        shadow_wrapper.addWidget(self.container)

        QTimer.singleShot(100, self._force_real_resize)

    def _force_real_resize(self):
        # forces a resize to the window because the fucking video widget doent display else wise and idk why
        current_size = self.size()
        self.resize(current_size.width() + 1, current_size.height() + 1)
        self.resize(current_size)  # back to original


    def container_resize_event(self, event):
        margin = 5
        x = self.container.width() - self.resize_grip.width() - margin
        y = self.container.height() - self.resize_grip.height() - margin
        self.resize_grip.move(x, y)
        event.accept()

    def _create_shadow(self):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(Qt.GlobalColor.black)
        return shadow

    def _build_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # custom title bar
        self.titlebar = TitleBar(self)
        layout.addWidget(self.titlebar)

        # main content layout
        content = QVBoxLayout()
        content.setContentsMargins(16, 16, 16, 16)
        layout.addLayout(content)

        # media area
        self.video_widget = QVideoWidget()
        
        self.image_label = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.hide()

        self.media_player = QMediaPlayer()
        self.media_player.mediaStatusChanged.connect(self.handle_media_status)
        self.media_player.setVideoOutput(self.video_widget)

        self.media_frame = QVBoxLayout()
        self.media_frame.addWidget(self.video_widget)
        self.media_frame.addWidget(self.image_label)
        content.addLayout(self.media_frame, stretch=3)

        # info boxes
        info_layout = QHBoxLayout()
        self.file_info = QTextEdit()
        self.file_info.setReadOnly(True)
        self.file_info.setPlaceholderText("File info will appear here")

        self.scan_output = QTextEdit()
        self.scan_output.setReadOnly(True)
        self.scan_output.setPlaceholderText("Scan output will appear here")

        info_layout.addWidget(self.file_info)
        info_layout.addWidget(self.scan_output)
        content.addLayout(info_layout, stretch=2)

        # buttons
        button_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.clicked.connect(self.select_file)
        button_layout.addWidget(self.upload_btn)

        self.start_btn = QPushButton("Start")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.toggle_start)
        button_layout.addWidget(self.start_btn)

        content.addLayout(button_layout)

        return layout

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image or Video",
                                              filter="Media Files (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)")
        if not path:
            return

        self.file_path = path
        self.start_btn.setEnabled(True)
        self.scan_output.clear()
        self.file_info.clear()

        ext = os.path.splitext(path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg"]:
            self.display_image(path)
            self.show_file_info_image(path)
        elif ext in [".mp4", ".avi", ".mov"]:
            self.display_video(path)
            self.show_file_info_video(path)
        else:
            self.file_info.setText("Unsupported file format")


    def toggle_start(self):
        running = getattr(self, "_running", False)
        if running:
            self.stop_realtime()
        else:
            self.start_realtime()

    def start_realtime(self):
        if not self.file_path:
            return
        # switch preview to QLabel (so we can draw our own frames)
        self.media_player.stop()
        self.video_widget.hide()
        self.image_label.show()

        self.upload_btn.setEnabled(False)
        self.start_btn.setText("Stop")
        self._running = True

        # spin up worker
        self.worker = InferenceThread(self.file_path, stride=1, ema_alpha=0.20, conf=0.25, parent=self)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.hud_text.connect(self.on_hud_text)
        self.worker.finished.connect(self.on_infer_finished)
        self.worker.start()

    def stop_realtime(self):
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        self._running = False
        self.start_btn.setText("Start")
        self.upload_btn.setEnabled(True)

    def on_frame_ready(self, qimg: QImage):
        # scale to fit label
        pix = QPixmap.fromImage(qimg).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    def on_hud_text(self, text: str):
        # show rolling risk in the right-hand box
        self.scan_output.setPlainText(text)

    def on_infer_finished(self, res: dict):
        self._running = False
        self.start_btn.setText("Start")
        self.upload_btn.setEnabled(True)
        # show final verdict + where the saved video is
        if "error" in res:
            self.scan_output.setPlainText(res["error"])
            return
        summary = f"{res['text']}\nSaved: {res['output_video']}\nBackend: {res.get('backend','?')}"
        self.scan_output.setPlainText(summary)


    def display_image(self, path):
        self.media_player.stop()
        self.video_widget.hide()
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.image_label.setPixmap(pixmap)
        self.image_label.show()

    def display_video(self, path):
        self.image_label.hide()
        self.video_widget.show()

        self.media_player.setSource(QUrl.fromLocalFile(path))
        self.media_player.play()
        
    def show_file_info_image(self, path):
        from PIL import Image
        img = Image.open(path)
        info = (
            f"Type: Image\n"
            f"Format: {img.format}\n"
            f"Size: {img.width} x {img.height}\n"
            f"Mode: {img.mode}\n"
        )
        self.file_info.setText(info)

    def show_file_info_video(self, path):
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frames / fps if fps else 0
        info = (
            f"Type: Video\n"
            f"Resolution: {width} x {height}\n"
            f"FPS: {fps:.2f}\n"
            f"Duration: {duration:.2f} sec\n"
            f"Frame Count: {frames}\n"
            f"Extension: {os.path.splitext(path)[-1]}\n"
        )
        self.file_info.setText(info)
        cap.release()

    # # start ur magic from here
    # def scan_file(self):
    #     self.scan_btn.setEnabled(False)
    #     self.upload_btn.setEnabled(False)
    #     # dummy model response
    #     result = model.placeholder_function(self.file_path)
    #     self.scan_output.setText(result)
        
    #     self.scan_btn.setEnabled(True)
    #     self.upload_btn.setEnabled(True)

    def handle_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.media_player.setPosition(0) 
            self.media_player.play()     


    def update_resize_grip_pos(self):
        margin = 5
        x = self.resize_grip.parent().width() - self.resize_grip.width() - margin
        y = self.resize_grip.parent().height() - self.resize_grip.height() - margin
        self.resize_grip.move(x, y)


class ResizeGrip(QWidget):
    def __init__(self, parent=None, size=16):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.SizeFDiagCursor)
        self._resizing = False
        self._start_pos = QPoint(0, 0)
        self._start_size = parent.size()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._resizing = True
            self._start_pos = event.globalPosition().toPoint()
            self._start_size = self.parent().size()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._resizing:
            current_pos = event.globalPosition().toPoint()
            diff = current_pos - self._start_pos
            new_width = max(self._start_size.width() + diff.x(), self.minimumWidth())
            new_height = max(self._start_size.height() + diff.y(), self.minimumHeight())
            self.window().resize(new_width, new_height)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._resizing = False
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        painter.setPen(QColor("#555555"))
        step = 4
        for i in range(0, self.width(), step):
            painter.drawLine(i, self.height(), self.width(), self.height() - i)

