"""
Experiment Dashboard — a PySide6 result browser for the analysis framework.

Design goals:
    * A modern dark-mode GUI consistent with the live demo look & feel.
    * Clean navigation: left-hand experiment picker, right-hand detail pane.
    * Metric summary cards + tabbed views for figures / tables / report / log.
    * Safe fallback: if PySide6 is not available, the entry point prints a
      diagnostic and exits cleanly.

Launch with:

    python -m gui.experiment_dashboard
    # or
    python scripts/launch_dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from PySide6.QtCore import Qt, QUrl
    from PySide6.QtGui import QColor, QFont, QIcon, QPalette, QPixmap, QDesktopServices
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QScrollArea,
        QSplitter,
        QStatusBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    PYSIDE_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYSIDE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_runs(results_root: Path) -> List[Dict[str, Any]]:
    """Scan ``results/`` for directories that look like experiment runs."""
    if not results_root.exists():
        return []
    runs: List[Dict[str, Any]] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        manifest = child / "manifest.json"
        summary = child / "summary.json"
        report = child / "report.md"
        if not manifest.exists() and not summary.exists() and not report.exists():
            continue
        runs.append({
            "name": child.name,
            "path": child,
            "has_manifest": manifest.exists(),
            "has_summary": summary.exists(),
            "has_report": report.exists(),
        })
    return runs


def load_summary(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            out["summary"] = json.loads(summary.read_text(encoding="utf-8"))
        except Exception as exc:
            out["summary"] = {"_error": str(exc)}
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            out["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            out["manifest"] = {"_error": str(exc)}
    return out


# ---------------------------------------------------------------------------
# Stylesheet (dark, polished)
# ---------------------------------------------------------------------------


STYLESHEET = """
QMainWindow, QWidget {
    background-color: #15181e;
    color: #e5e9f0;
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", "Arial", sans-serif;
    font-size: 13px;
}
QLabel#TitleLabel {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #f5f7fa;
}
QLabel#SubtitleLabel {
    color: #8a93a6;
    font-size: 12px;
}
QListWidget {
    background-color: #1b1f27;
    border: 1px solid #252a34;
    border-radius: 8px;
    padding: 6px;
    outline: 0;
}
QListWidget::item {
    padding: 10px 12px;
    border-radius: 6px;
    color: #cfd6e4;
}
QListWidget::item:selected {
    background-color: #2f6fed;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background-color: #232834;
}
QFrame#Card {
    background-color: #1b1f27;
    border: 1px solid #252a34;
    border-radius: 10px;
}
QLabel#CardLabel {
    color: #8a93a6;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}
QLabel#CardValue {
    color: #f5f7fa;
    font-size: 18px;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #252a34;
    border-radius: 8px;
    top: -1px;
    background-color: #1b1f27;
}
QTabBar::tab {
    background-color: transparent;
    color: #8a93a6;
    padding: 7px 18px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected {
    background-color: #1b1f27;
    color: #f5f7fa;
    border-bottom: 2px solid #2f6fed;
}
QTabBar::tab:hover:!selected {
    color: #cfd6e4;
}
QTextEdit {
    background-color: #1b1f27;
    border: 1px solid #252a34;
    border-radius: 6px;
    color: #e5e9f0;
    selection-background-color: #2f6fed;
    font-family: "SF Mono", "JetBrains Mono", "Consolas", monospace;
    font-size: 12px;
}
QTableWidget {
    background-color: #1b1f27;
    border: 1px solid #252a34;
    border-radius: 6px;
    gridline-color: #252a34;
    selection-background-color: #2f6fed;
    selection-color: #ffffff;
    color: #e5e9f0;
}
QHeaderView::section {
    background-color: #232834;
    color: #cfd6e4;
    border: 0;
    padding: 6px;
    font-weight: 600;
}
QPushButton {
    background-color: #2f6fed;
    color: #ffffff;
    border: 0;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover { background-color: #5388f5; }
QPushButton:pressed { background-color: #254ead; }
QPushButton#SecondaryButton {
    background-color: #232834;
    color: #cfd6e4;
}
QPushButton#SecondaryButton:hover { background-color: #2b3141; }
QScrollArea { border: 0; background-color: transparent; }
QStatusBar { background-color: #1b1f27; color: #8a93a6; }
QStatusBar::item { border: none; }
"""


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


if PYSIDE_AVAILABLE:

    class MetricCard(QFrame):
        """A flat card displaying a single label + metric value."""

        def __init__(self, label: str, value: str, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setObjectName("Card")
            self.setFrameShape(QFrame.NoFrame)
            layout = QVBoxLayout(self)
            layout.setContentsMargins(14, 12, 14, 12)
            layout.setSpacing(4)
            lbl = QLabel(label)
            lbl.setObjectName("CardLabel")
            val = QLabel(value)
            val.setObjectName("CardValue")
            val.setWordWrap(True)
            layout.addWidget(lbl)
            layout.addWidget(val)

    class FigureGallery(QScrollArea):
        """Grid of figure thumbnails with click-to-open behaviour."""

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            self.setWidgetResizable(True)
            self._container = QWidget()
            self._grid = QGridLayout(self._container)
            self._grid.setContentsMargins(4, 4, 4, 4)
            self._grid.setSpacing(10)
            self.setWidget(self._container)

        def clear(self) -> None:
            while self._grid.count():
                item = self._grid.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        def set_images(self, paths: List[Path]) -> None:
            self.clear()
            cols = 2
            for i, path in enumerate(paths):
                card = QFrame()
                card.setObjectName("Card")
                cl = QVBoxLayout(card)
                cl.setContentsMargins(10, 10, 10, 10)
                title = QLabel(path.name)
                title.setObjectName("CardLabel")
                title.setWordWrap(True)
                pix = QPixmap(str(path))
                if not pix.isNull():
                    pix = pix.scaledToWidth(460, Qt.SmoothTransformation)
                lbl = QLabel()
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setPixmap(pix)
                btn = QPushButton("Open")
                btn.setObjectName("SecondaryButton")
                btn.clicked.connect(
                    lambda _checked=False, p=path: QDesktopServices.openUrl(
                        QUrl.fromLocalFile(str(p))
                    )
                )
                cl.addWidget(title)
                cl.addWidget(lbl)
                cl.addWidget(btn)
                self._grid.addWidget(card, i // cols, i % cols)

    class RunDetailView(QWidget):
        """Right-hand pane: header + metric cards + tabbed view."""

        def __init__(self, parent: Optional[QWidget] = None):
            super().__init__(parent)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(16, 16, 16, 16)
            outer.setSpacing(12)

            self._title = QLabel("Select an experiment run")
            self._title.setObjectName("TitleLabel")
            self._subtitle = QLabel("")
            self._subtitle.setObjectName("SubtitleLabel")
            outer.addWidget(self._title)
            outer.addWidget(self._subtitle)

            self._cards_host = QWidget()
            self._cards_layout = QGridLayout(self._cards_host)
            self._cards_layout.setContentsMargins(0, 6, 0, 0)
            self._cards_layout.setSpacing(10)
            outer.addWidget(self._cards_host)

            self._tabs = QTabWidget()
            outer.addWidget(self._tabs, stretch=1)

            self._report_edit = QTextEdit()
            self._report_edit.setReadOnly(True)
            self._log_edit = QTextEdit()
            self._log_edit.setReadOnly(True)
            self._summary_edit = QTextEdit()
            self._summary_edit.setReadOnly(True)
            self._tables_tab = QWidget()
            self._tables_layout = QVBoxLayout(self._tables_tab)
            self._tables_layout.setContentsMargins(0, 0, 0, 0)
            self._tables_layout.setSpacing(8)
            self._gallery = FigureGallery()

            self._tabs.addTab(self._gallery, "Figures")
            self._tabs.addTab(self._tables_tab, "Tables")
            self._tabs.addTab(self._report_edit, "Report")
            self._tabs.addTab(self._summary_edit, "Summary JSON")
            self._tabs.addTab(self._log_edit, "Log")

        # ----- helpers --------------------------------------------------
        def _clear_cards(self) -> None:
            while self._cards_layout.count():
                item = self._cards_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        def _clear_tables(self) -> None:
            while self._tables_layout.count():
                item = self._tables_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.deleteLater()

        # ----- data loading ---------------------------------------------
        def show_empty(self) -> None:
            self._title.setText("Select an experiment run")
            self._subtitle.setText("")
            self._clear_cards()
            self._clear_tables()
            self._report_edit.clear()
            self._log_edit.clear()
            self._summary_edit.clear()
            self._gallery.clear()

        def show_run(self, run: Dict[str, Any]) -> None:
            run_dir: Path = run["path"]
            payload = load_summary(run_dir)
            summary = payload.get("summary") or {}
            manifest = payload.get("manifest") or {}
            experiment = manifest.get("experiment") or summary.get("experiment") or run_dir.name
            self._title.setText(run_dir.name)
            extra = manifest.get("extra", {}).get("run", {}) if isinstance(manifest, dict) else {}
            subtitle_parts = [f"experiment = {experiment}"]
            if extra.get("started_at"):
                subtitle_parts.append(f"started {extra['started_at']}")
            if extra.get("finished_at"):
                subtitle_parts.append(f"finished {extra['finished_at']}")
            self._subtitle.setText("   •   ".join(subtitle_parts))

            # Metric cards — heuristic extraction
            self._clear_cards()
            cards = self._extract_cards(summary)
            for i, (k, v) in enumerate(cards):
                self._cards_layout.addWidget(MetricCard(k, v), i // 4, i % 4)

            # Figures
            figures_dir = run_dir / "figures"
            images = sorted(figures_dir.glob("*.png")) if figures_dir.exists() else []
            self._gallery.set_images(images)

            # Tables
            self._clear_tables()
            tables_dir = run_dir / "tables"
            if tables_dir.exists():
                for csv_path in sorted(tables_dir.glob("*.csv")):
                    self._tables_layout.addWidget(self._build_table_card(csv_path))
            self._tables_layout.addStretch(1)

            # Report
            report_path = run_dir / "report.md"
            self._report_edit.setPlainText(
                report_path.read_text(encoding="utf-8") if report_path.exists() else "(no report.md)"
            )
            # Log
            log_path = run_dir / "run.log"
            self._log_edit.setPlainText(
                log_path.read_text(encoding="utf-8") if log_path.exists() else "(no run.log)"
            )
            # Summary
            self._summary_edit.setPlainText(
                json.dumps(summary, indent=2) if summary else "(no summary.json)"
            )

        @staticmethod
        def _extract_cards(summary: Dict[str, Any]) -> List[tuple[str, str]]:
            if not isinstance(summary, dict):
                return []
            out: List[tuple[str, str]] = []

            def _add(label: str, value: Any) -> None:
                if value is None:
                    return
                if isinstance(value, float):
                    out.append((label, f"{value:.4g}"))
                elif isinstance(value, (int, str)):
                    out.append((label, str(value)))

            _add("Captures", summary.get("n_captures"))
            _add("Features", summary.get("n_features"))
            _add("Fibers", summary.get("n_fibers"))
            _add("Challenges", summary.get("n_challenges"))

            for section_key in ("verification", "known_challenge", "unknown_challenge"):
                section = summary.get(section_key)
                if isinstance(section, dict):
                    for metric, label in (
                        ("top1_accuracy", "Top-1"),
                        ("auc", "AUC"),
                        ("eer", "EER"),
                    ):
                        if metric in section:
                            _add(f"{section_key.replace('_', ' ').title()} · {label}",
                                 section[metric])

            pf = summary.get("power_fluctuation")
            if isinstance(pf, dict):
                _add("Mean CV (green)", pf.get("mean_cv_green"))
                _add("Mean CV (ratio)", pf.get("mean_cv_ratio"))
                _add("Reduction factor", pf.get("mean_reduction_factor"))

            ri = summary.get("reinstall")
            if isinstance(ri, dict):
                _add("Mean NCC (green)", ri.get("mean_within_ncc_green"))
                _add("Mean NCC (ratio)", ri.get("mean_within_ncc_ratio"))

            return out[:8]

        def _build_table_card(self, csv_path: Path) -> QWidget:
            container = QFrame()
            container.setObjectName("Card")
            layout = QVBoxLayout(container)
            layout.setContentsMargins(10, 10, 10, 10)
            header = QLabel(csv_path.name)
            header.setObjectName("CardLabel")
            layout.addWidget(header)
            table = QTableWidget()
            table.setEditTriggers(QTableWidget.NoEditTriggers)
            table.horizontalHeader().setStretchLastSection(True)
            try:
                import csv
                with csv_path.open("r", encoding="utf-8") as f:
                    reader = list(csv.reader(f))
                if reader:
                    headers = reader[0]
                    rows = reader[1:]
                    table.setColumnCount(len(headers))
                    table.setHorizontalHeaderLabels(headers)
                    table.setRowCount(min(len(rows), 200))
                    for i, row in enumerate(rows[:200]):
                        for j, cell in enumerate(row):
                            item = QTableWidgetItem(cell)
                            table.setItem(i, j, item)
            except Exception as exc:
                table.setRowCount(1)
                table.setColumnCount(1)
                table.setItem(0, 0, QTableWidgetItem(f"Failed to load {csv_path.name}: {exc}"))
            layout.addWidget(table)
            return container

    class ExperimentDashboard(QMainWindow):
        def __init__(self, results_root: Path):
            super().__init__()
            self.results_root = results_root
            self.setWindowTitle("Speckle-PUF Experiment Dashboard")
            self.resize(1280, 820)
            self.setStyleSheet(STYLESHEET)

            central = QWidget()
            self.setCentralWidget(central)
            outer = QHBoxLayout(central)
            outer.setContentsMargins(16, 16, 16, 16)
            outer.setSpacing(16)

            # Left: experiment list
            left = QWidget()
            left_layout = QVBoxLayout(left)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(8)

            title = QLabel("Experiment Runs")
            title.setObjectName("TitleLabel")
            subtitle = QLabel(f"{results_root}")
            subtitle.setObjectName("SubtitleLabel")
            subtitle.setWordWrap(True)

            self._list = QListWidget()
            self._list.setMinimumWidth(280)
            self._list.itemSelectionChanged.connect(self._on_selection)

            btn_row = QHBoxLayout()
            refresh_btn = QPushButton("Refresh")
            refresh_btn.setObjectName("SecondaryButton")
            refresh_btn.clicked.connect(self.refresh)
            open_btn = QPushButton("Open folder…")
            open_btn.setObjectName("SecondaryButton")
            open_btn.clicked.connect(self._choose_folder)
            btn_row.addWidget(refresh_btn)
            btn_row.addWidget(open_btn)
            btn_row.addStretch(1)

            left_layout.addWidget(title)
            left_layout.addWidget(subtitle)
            left_layout.addWidget(self._list, stretch=1)
            left_layout.addLayout(btn_row)

            # Right: detail view
            self._detail = RunDetailView()

            splitter = QSplitter(Qt.Horizontal)
            splitter.addWidget(left)
            splitter.addWidget(self._detail)
            splitter.setStretchFactor(0, 0)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([320, 960])
            outer.addWidget(splitter)

            self._status = QStatusBar()
            self.setStatusBar(self._status)
            self.refresh()

        # ----- actions ----------------------------------------------------
        def refresh(self) -> None:
            self._list.clear()
            runs = discover_runs(self.results_root)
            for run in runs:
                item = QListWidgetItem(run["name"])
                item.setData(Qt.UserRole, run)
                self._list.addItem(item)
            self._status.showMessage(f"{len(runs)} runs found in {self.results_root}")
            if runs:
                self._list.setCurrentRow(0)
            else:
                self._detail.show_empty()

        def _choose_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(
                self, "Choose results directory", str(self.results_root)
            )
            if folder:
                self.results_root = Path(folder)
                self.refresh()

        def _on_selection(self) -> None:
            items = self._list.selectedItems()
            if not items:
                self._detail.show_empty()
                return
            run = items[0].data(Qt.UserRole)
            self._detail.show_run(run)

    def _apply_dark_palette(app: "QApplication") -> None:
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(21, 24, 30))
        palette.setColor(QPalette.WindowText, QColor(229, 233, 240))
        palette.setColor(QPalette.Base, QColor(27, 31, 39))
        palette.setColor(QPalette.AlternateBase, QColor(35, 40, 52))
        palette.setColor(QPalette.ToolTipBase, QColor(229, 233, 240))
        palette.setColor(QPalette.ToolTipText, QColor(21, 24, 30))
        palette.setColor(QPalette.Text, QColor(229, 233, 240))
        palette.setColor(QPalette.Button, QColor(35, 40, 52))
        palette.setColor(QPalette.ButtonText, QColor(229, 233, 240))
        palette.setColor(QPalette.Highlight, QColor(47, 111, 237))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        app.setPalette(palette)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    if not PYSIDE_AVAILABLE:
        print("[ERROR] PySide6 is not installed.\n  Install with:  pip install PySide6")
        return 2
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    argv = list(argv) if argv is not None else list(sys.argv[1:])
    results_root = Path(argv[0]) if argv else Path(__file__).resolve().parents[1] / "results"

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Speckle-PUF Experiment Dashboard")
    app.setStyle("Fusion")
    _apply_dark_palette(app)  # type: ignore[name-defined]

    window = ExperimentDashboard(results_root)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
