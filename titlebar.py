from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import QWidget, QLabel, QPushButton, QHBoxLayout

class TitleBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet("background-color: #1e1e1e; color: white;")

        self.start = QPoint(0, 0)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)

        self.title = QLabel("Suspicious Behavior Detector")
        self.title.setStyleSheet("font-weight: bold; padding-left: 8px; padding-right: 8px;")
        layout.addWidget(self.title)
        layout.addStretch()

        self.btn_close = QPushButton("✕")
        self.btn_close.setFixedSize(28, 28)
        self.btn_close.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: red;
            }
        """)
        self.btn_close.clicked.connect(self.window().close)
        layout.addWidget(self.btn_close)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            diff = event.globalPosition().toPoint() - self.start
            self.window().move(self.window().pos() + diff)
            self.start = event.globalPosition().toPoint()
