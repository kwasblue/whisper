import sys
from PySide6 import QtWidgets
from mainwindow import MainWindow
from styles import apply_theme

def main():
    """Entry point for Whisper Notebook."""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_theme(app)

    win = MainWindow()
    win.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
