# Video Preprocessor - FOI Extractor

This Windows application automatically preprocesses video files captured by equipment and stored in a specific folder. It extracts only the specified FOI (Field of Interest) region, discards unnecessary parts, and saves compressed videos to reduce file size.

## Features

- **Automatic Monitoring**: Monitors a specified folder for new video files.
- **ROI Extraction**: Extracts only the region of interest from each frame.
- **Compression**: Saves processed videos in MP4 format to reduce file size.
- **GUI Interface**: User-friendly interface for setting folders and ROI parameters.
- **Cross-Platform**: Developed in Python, can be built for Windows using PyInstaller.

## Requirements

- Python 3.8+
- OpenCV
- PyQt5
- Watchdog

Install dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Run `main.py` to start the application.
2. Select the input folder to monitor for new videos.
3. Select the output folder for processed videos.
4. Set the ROI coordinates (x, y, width, height).
5. Click "Start Monitoring" to begin automatic processing.

## Building for Windows

Use PyInstaller to create a standalone executable:

```
pip install pyinstaller
pyinstaller --onefile --windowed main.py
```

This will generate `dist/main.exe` which can be run on Windows.

## Supported Video Formats

- MP4
- AVI
- MOV

(Add more formats by modifying the file extension check in `gui.py`)

## License

This project is part of the DREAM-SPTP research initiative.