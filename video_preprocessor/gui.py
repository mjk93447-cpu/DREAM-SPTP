import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QLineEdit, QFileDialog, QTextEdit,
                             QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import video_processor

class VideoProcessorHandler(FileSystemEventHandler):
    def __init__(self, input_folder, output_folder, roi, log_callback):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.roi = roi
        self.log_callback = log_callback

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(('.mp4', '.avi', '.mov')):  # Add more extensions if needed
            self.log_callback(f"New video detected: {event.src_path}")
            filename = os.path.basename(event.src_path)
            output_path = os.path.join(self.output_folder, f"processed_{filename}")
            try:
                video_processor.process_video(event.src_path, output_path, self.roi, delete_original=True)
                self.log_callback(f"Processed and saved: {output_path}")
                self.log_callback(f"Deleted original file: {event.src_path}")
            except Exception as e:
                self.log_callback(f"Error processing {event.src_path}: {str(e)}")

class MonitorThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, input_folder, output_folder, roi):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.roi = roi
        self.observer = None

    def run(self):
        event_handler = VideoProcessorHandler(self.input_folder, self.output_folder, self.roi, self.log_signal.emit)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.input_folder, recursive=False)
        self.observer.start()
        self.log_signal.emit(f"Monitoring started for folder: {self.input_folder}")
        self.observer.join()

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.log_signal.emit("Monitoring stopped.")

class VideoPreprocessorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.monitor_thread = None

    def initUI(self):
        self.setWindowTitle('Video Preprocessor - FOI Extractor')
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        # Input folder selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel('Input Folder:'))
        self.input_folder_edit = QLineEdit()
        input_layout.addWidget(self.input_folder_edit)
        self.input_folder_btn = QPushButton('Browse')
        self.input_folder_btn.clicked.connect(self.select_input_folder)
        input_layout.addWidget(self.input_folder_btn)
        layout.addLayout(input_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel('Output Folder:'))
        self.output_folder_edit = QLineEdit()
        output_layout.addWidget(self.output_folder_edit)
        self.output_folder_btn = QPushButton('Browse')
        self.output_folder_btn.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_folder_btn)
        layout.addLayout(output_layout)

        # ROI settings
        roi_layout = QHBoxLayout()
        roi_layout.addWidget(QLabel('ROI (x,y,w,h):'))
        self.roi_x_edit = QLineEdit('0')
        self.roi_y_edit = QLineEdit('0')
        self.roi_w_edit = QLineEdit('640')
        self.roi_h_edit = QLineEdit('480')
        roi_layout.addWidget(self.roi_x_edit)
        roi_layout.addWidget(self.roi_y_edit)
        roi_layout.addWidget(self.roi_w_edit)
        roi_layout.addWidget(self.roi_h_edit)
        layout.addLayout(roi_layout)

        # Start/Stop buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton('Start Monitoring')
        self.start_btn.clicked.connect(self.start_monitoring)
        button_layout.addWidget(self.start_btn)
        self.stop_btn = QPushButton('Stop Monitoring')
        self.stop_btn.clicked.connect(self.stop_monitoring)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        # Log area
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        self.setLayout(layout)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Input Folder')
        if folder:
            self.input_folder_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder:
            self.output_folder_edit.setText(folder)

    def start_monitoring(self):
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        if not input_folder or not output_folder:
            QMessageBox.warning(self, 'Error', 'Please select both input and output folders.')
            return
        if not os.path.exists(input_folder):
            QMessageBox.warning(self, 'Error', 'Input folder does not exist.')
            return
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        try:
            roi = (int(self.roi_x_edit.text()), int(self.roi_y_edit.text()),
                   int(self.roi_w_edit.text()), int(self.roi_h_edit.text()))
        except ValueError:
            QMessageBox.warning(self, 'Error', 'Invalid ROI values. Please enter integers.')
            return

        self.monitor_thread = MonitorThread(input_folder, output_folder, roi)
        self.monitor_thread.log_signal.connect(self.update_log)
        self.monitor_thread.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_monitoring(self):
        if self.monitor_thread:
            self.monitor_thread.stop()
            self.monitor_thread = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_log(self, message):
        self.log_edit.append(message)

    def closeEvent(self, event):
        self.stop_monitoring()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = VideoPreprocessorGUI()
    gui.show()
    sys.exit(app.exec_())