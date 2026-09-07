from __future__ import annotations

import ctypes
import importlib
import logging
import logging.handlers
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

from PySide6.QtCore import QRectF, QStandardPaths, Qt, qInstallMessageHandler
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QSplashScreen

from . import __version__

WINDOWS_APP_ID = "ToolBox.StereoSelector"
UI_FONT_CANDIDATES = (
    "Microsoft YaHei UI",  # Windows, CJK capable
    "PingFang SC",  # macOS
    "Noto Sans CJK SC",  # Linux
    "Segoe UI",
    "Helvetica Neue",
    "DejaVu Sans",
)

logger = logging.getLogger(__name__)


def asset_path(name: str) -> Path:
    return Path(__file__).with_name("assets") / name


def application_icon() -> QIcon:
    return QIcon(str(asset_path("app_icon.png")))


def preferred_font_family() -> str:
    """Return the first CJK-capable UI family available on this machine."""
    families = set(QFontDatabase.families())
    return next((name for name in UI_FONT_CANDIDATES if name in families), QFont().family())


def application_font() -> QFont:
    """Use one CJK-capable UI family for Qt widgets and native popup surfaces."""
    font = QFont(preferred_font_family())
    font.setPointSizeF(9.5)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    return font


def log_directory() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppLocalDataLocation)
    if not base:
        base = str(Path.home() / ".stereo_selector")
    directory = Path(base)
    if directory.name != "StereoSelector":
        directory = directory / "ToolBox" / "StereoSelector"
    return directory / "logs"


def configure_logging(verbose: bool = False) -> Path | None:
    """Write application logs to the user data folder and mirror them to stderr.

    Packaged builds run without a console, so this file is the only place a
    failed load, a dropped background task or a Qt warning can be recovered.
    """
    root = logging.getLogger()
    if any(getattr(handler, "_stereo_selector", False) for handler in root.handlers):
        return getattr(root, "_stereo_selector_log_path", None)
    level = logging.DEBUG if verbose or os.environ.get("STEREO_SELECTOR_DEBUG") else logging.INFO
    root.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream._stereo_selector = True  # type: ignore[attr-defined]
    root.addHandler(stream)
    path: Path | None = None
    try:
        directory = log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "stereo_selector.log"
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        handler._stereo_selector = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    except OSError as exc:
        logger.warning("无法创建日志文件: %s", exc)
    root._stereo_selector_log_path = path  # type: ignore[attr-defined]

    def qt_message(mode, context, message) -> None:
        logging.getLogger("qt").log(
            {0: logging.DEBUG, 1: logging.WARNING, 2: logging.CRITICAL, 3: logging.CRITICAL, 4: logging.INFO}.get(
                int(mode.value) if hasattr(mode, "value") else int(mode), logging.INFO
            ),
            "%s",
            message,
        )

    qInstallMessageHandler(qt_message)
    logger.info("Stereo Selector %s 启动，日志文件：%s", __version__, path)
    return path


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

    family = preferred_font_family()
    painter.setPen(QColor("#f2f2f2"))
    title_font = QFont(family, 18)
    title_font.setWeight(QFont.Weight.DemiBold)
    painter.setFont(title_font)
    painter.drawText(126, 72, "Stereo Selector")

    painter.setPen(QColor("#929292"))
    painter.setFont(QFont(family, 10))
    painter.drawText(126, 99, "双目图像与点云查看器")

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
        logger.exception("点云后端预加载失败")


def main(argv: list[str] | None = None) -> int:
    configure_logging(verbose="--verbose" in (argv if argv is not None else sys.argv[1:]))
    _set_windows_app_id()
    qt_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication(qt_argv)
    app.setApplicationName("Stereo Selector")
    app.setOrganizationName("ToolBox")
    app.setApplicationVersion(__version__)
    app.setFont(application_font())
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
        logger.exception("应用启动失败")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
