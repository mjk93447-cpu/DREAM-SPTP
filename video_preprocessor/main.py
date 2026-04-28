#!/usr/bin/env python3
"""
Video Preprocessor Application
Automatically processes video files by extracting FOI (Field of Interest) regions
to reduce file size and focus on relevant areas.
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from gui import VideoPreprocessorGUI
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = VideoPreprocessorGUI()
    gui.show()
    sys.exit(app.exec_())