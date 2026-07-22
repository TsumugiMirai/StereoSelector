from __future__ import annotations

import ctypes
import importlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from . import __version__


WINDOWS_APP_ID = "ToolBox.StereoSelector"


def asset_path(name: str) -> Path:
    return Path(__file__).with_name("assets") / name


def application_icon() -> QIcon:
    return QIcon(str(asset_path("app_icon.png")))


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)
    except (AttributeError, OSError):
        pass


def _splash_pixmap(icon: QIcon) -> QPixmap:
    pixmap = QPixmap(500, 228)
    pixmap.fill(QColor("#101010"))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    painter.setPen(QPen(QColor("#343434"), 1))
    painter.setBrush(QColor("#171717"))
    panel = QPainterPath()
    panel.addRoundedRect(QRectF(1, 1, 498, 226), 14, 14)
    painter.drawPath(panel)

    icon_pixmap = icon.pixmap(64, 64)
    painter.drawPixmap(42, 48, icon_pixmap)

    painter.setPen(QColor("#f2f2f2"))
    title_font = QFont("Segoe UI", 18)
    title_font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(title_font)
    painter.drawText(126, 72, "Stereo Selector")

    painter.setPen(QColor("#929292"))
    painter.setFont(QFont("Microsoft YaHei UI", 10))
    painter.drawText(126, 99, "双目图像与点云筛选器")

    painter.setPen(QPen(QColor("#303030"), 1))
    painter.drawLine(42, 137, 458, 137)
    painter.setPen(QColor("#b7b7b7"))
    painter.drawText(42, 178, "正在启动…")
    painter.end()
    return pixmap


def _preload_point_cloud_backend() -> None:
    try:
        importlib.import_module("pyqtgraph.opengl")
    except Exception:
        # PointCloudCanvas presents a useful in-app error if OpenGL is unavailable.
        pass


def main(argv: list[str] | None = None) -> int:
    _set_windows_app_id()
    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication(qt_argv)
    app.setApplicationName("Stereo Selector")
    app.setOrganizationName("ToolBox")
    app.setApplicationVersion(__version__)
    icon = application_icon()
    app.setWindowIcon(icon)

    splash = QSplashScreen(_splash_pixmap(icon), Qt.WindowType.WindowStaysOnTopHint)
    splash.setWindowIcon(icon)
    splash.show()
    app.processEvents()

    try:
        # Import the heavy OpenGL backend alongside the rest of the application.
        # The event pump keeps the already-painted splash responsive while the
        # final backend import completes.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="stereo-startup") as executor:
            backend_ready = executor.submit(_preload_point_cloud_backend)
            from .app import run_application

            while not backend_ready.done():
                app.processEvents()
                try:
                    backend_ready.result(timeout=0.02)
                except TimeoutError:
                    pass

        return run_application(app, argv, splash)
    except BaseException:
        splash.close()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
