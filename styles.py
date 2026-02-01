"""
Centralized styling for Whisper Notebook.

Provides a dark theme with purple/violet accent colors.
"""

# Color Palette
COLORS = {
    "bg_dark": "#1e1e2e",
    "bg_surface": "#2d2d3d",
    "bg_hover": "#3d3d4d",
    "border": "#3d3d4d",
    "text_primary": "#e0e0e0",
    "text_secondary": "#a0a0b0",
    "text_disabled": "#606070",
    "accent": "#9d7cd8",
    "accent_hover": "#b392f0",
    "accent_pressed": "#8668c4",
    "success": "#73daca",
    "warning": "#ffa657",
    "error": "#f7768e",
    "error_hover": "#ff8fa3",
}

DARK_STYLESHEET = f"""
/* ===== Global ===== */
QWidget {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_primary"]};
    font-family: -apple-system, "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: {COLORS["bg_surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px 16px;
    min-width: 80px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS["bg_hover"]};
    border-color: {COLORS["accent"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["accent_pressed"]};
}}

QPushButton:disabled {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_disabled"]};
    border-color: {COLORS["bg_surface"]};
}}

QPushButton:checked {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
}}

/* Primary action buttons */
QPushButton#startBtn, QPushButton#loadBtn {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
    color: white;
}}

QPushButton#startBtn:hover, QPushButton#loadBtn:hover {{
    background-color: {COLORS["accent_hover"]};
    border-color: {COLORS["accent_hover"]};
}}

QPushButton#startBtn:disabled, QPushButton#loadBtn:disabled {{
    background-color: {COLORS["bg_surface"]};
    color: {COLORS["text_disabled"]};
}}

/* Stop/Cancel buttons */
QPushButton#stopBtn, QPushButton#cancelBtn {{
    background-color: {COLORS["bg_surface"]};
    border-color: {COLORS["error"]};
    color: {COLORS["error"]};
}}

QPushButton#stopBtn:hover, QPushButton#cancelBtn:hover {{
    background-color: {COLORS["error"]};
    color: white;
}}

QPushButton#stopBtn:disabled, QPushButton#cancelBtn:disabled {{
    border-color: {COLORS["bg_surface"]};
    color: {COLORS["text_disabled"]};
}}

/* ===== Text Edit ===== */
QTextEdit {{
    background-color: {COLORS["bg_surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 8px;
    font-family: "SF Mono", "Menlo", "Monaco", "Consolas", monospace;
    font-size: 12px;
    selection-background-color: {COLORS["accent"]};
}}

/* ===== Table ===== */
QTableWidget {{
    background-color: {COLORS["bg_surface"]};
    alternate-background-color: {COLORS["bg_dark"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    gridline-color: {COLORS["border"]};
    selection-background-color: {COLORS["accent"]};
}}

QTableWidget::item {{
    padding: 6px;
    border: none;
}}

QTableWidget::item:selected {{
    background-color: {COLORS["accent"]};
    color: white;
}}

QTableWidget::item:hover {{
    background-color: {COLORS["bg_hover"]};
}}

QHeaderView::section {{
    background-color: {COLORS["bg_dark"]};
    color: {COLORS["text_secondary"]};
    padding: 8px;
    border: none;
    border-bottom: 1px solid {COLORS["border"]};
    font-weight: 600;
}}

/* ===== Progress Bar ===== */
QProgressBar {{
    background-color: {COLORS["bg_surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    text-align: center;
    color: {COLORS["text_primary"]};
    height: 20px;
}}

QProgressBar::chunk {{
    background-color: {COLORS["accent"]};
    border-radius: 3px;
}}

/* ===== Splitter ===== */
QSplitter::handle {{
    background-color: {COLORS["border"]};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {COLORS["accent"]};
}}

/* ===== Scrollbar ===== */
QScrollBar:vertical {{
    background-color: {COLORS["bg_dark"]};
    width: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS["bg_hover"]};
    border-radius: 5px;
    min-height: 30px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS["accent"]};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS["bg_dark"]};
    height: 10px;
    border-radius: 5px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS["bg_hover"]};
    border-radius: 5px;
    min-width: 30px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {COLORS["accent"]};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* ===== Labels ===== */
QLabel {{
    background-color: transparent;
    color: {COLORS["text_primary"]};
}}

QLabel#titleLabel {{
    font-size: 14px;
    font-weight: 600;
}}

/* ===== File Dialog ===== */
QFileDialog {{
    background-color: {COLORS["bg_dark"]};
}}
"""


def apply_theme(app):
    """Apply the dark theme to the application."""
    app.setStyleSheet(DARK_STYLESHEET)
