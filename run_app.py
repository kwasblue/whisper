import sys
from PySide6 import QtWidgets
from mainwindow import MainWindow

def main():
    """Entry point for Whisper Notebook."""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")

    win = MainWindow()
    win.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
