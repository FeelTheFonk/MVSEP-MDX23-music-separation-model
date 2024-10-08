import sys
import os
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from inference import predict_with_model, __VERSION__
import torch

root = {}

class AudioFileModel(QAbstractListModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_files = []

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return os.path.basename(self.audio_files[index.row()])
        elif role == Qt.ItemDataRole.ToolTipRole:
            return self.audio_files[index.row()]

    def rowCount(self, index):
        return len(self.audio_files)

    def add_files(self, files):
        self.beginInsertRows(QModelIndex(), len(self.audio_files), len(self.audio_files) + len(files) - 1)
        self.audio_files.extend(files)
        self.endInsertRows()

    def remove_files(self, indices):
        indices.sort(reverse=True)
        for index in indices:
            self.beginRemoveRows(QModelIndex(), index, index)
            del self.audio_files[index]
            self.endRemoveRows()

class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, options):
        super().__init__()
        self.options = options
        self._is_running = True

    def run(self):
        try:
            self.options['update_percent_func'] = self.update_progress
            predict_with_model(self.options)
            if self._is_running:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def update_progress(self, percent):
        if self._is_running:
            self.progress.emit(percent)

    def stop(self):
        self._is_running = False

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(400, 350)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()
        self.checkbox_cpu = QCheckBox("Use CPU instead of GPU")
        self.checkbox_single_onnx = QCheckBox("Use single ONNX")
        self.checkbox_large_gpu = QCheckBox("Use large GPU")
        self.checkbox_kim_1 = QCheckBox("Use old Kim Vocal model")
        self.checkbox_only_vocals = QCheckBox("Generate only vocals/instrumental")

        self.chunk_size = QLineEdit()
        self.chunk_size.setValidator(QIntValidator(100000, 10000000))
        self.overlap_large = QLineEdit()
        self.overlap_large.setValidator(QDoubleValidator(0.001, 0.999, 3))
        self.overlap_small = QLineEdit()
        self.overlap_small.setValidator(QDoubleValidator(0.001, 0.999, 3))

        form_layout = QFormLayout()
        form_layout.addRow("Chunk size:", self.chunk_size)
        form_layout.addRow("Overlap large:", self.overlap_large)
        form_layout.addRow("Overlap small:", self.overlap_small)

        button_layout = QHBoxLayout()
        self.button_save = QPushButton("Save settings")
        self.button_cancel = QPushButton("Cancel")
        button_layout.addWidget(self.button_save)
        button_layout.addWidget(self.button_cancel)

        layout.addWidget(self.checkbox_cpu)
        layout.addWidget(self.checkbox_single_onnx)
        layout.addWidget(self.checkbox_large_gpu)
        layout.addWidget(self.checkbox_kim_1)
        layout.addWidget(self.checkbox_only_vocals)
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.button_save.clicked.connect(self.save_settings)
        self.button_cancel.clicked.connect(self.reject)

    def load_settings(self):
        self.checkbox_cpu.setChecked(root['cpu'])
        self.checkbox_single_onnx.setChecked(root['single_onnx'])
        self.checkbox_large_gpu.setChecked(root['large_gpu'])
        self.checkbox_kim_1.setChecked(root['use_kim_model_1'])
        self.checkbox_only_vocals.setChecked(root['only_vocals'])
        self.chunk_size.setText(str(root['chunk_size']))
        self.overlap_large.setText(str(root['overlap_large']))
        self.overlap_small.setText(str(root['overlap_small']))

    def save_settings(self):
        try:
            chunk_size = int(self.chunk_size.text())
            overlap_large = float(self.overlap_large.text())
            overlap_small = float(self.overlap_small.text())
            if not (100000 <= chunk_size <= 10000000):
                raise ValueError("Chunk size must be between 100,000 and 10,000,000")
            if not (0.001 <= overlap_large <= 0.999):
                raise ValueError("Overlap large must be between 0.001 and 0.999")
            if not (0.001 <= overlap_small <= 0.999):
                raise ValueError("Overlap small must be between 0.001 and 0.999")

            root['cpu'] = self.checkbox_cpu.isChecked()
            root['single_onnx'] = self.checkbox_single_onnx.isChecked()
            root['large_gpu'] = self.checkbox_large_gpu.isChecked()
            root['use_kim_model_1'] = self.checkbox_kim_1.isChecked()
            root['only_vocals'] = self.checkbox_only_vocals.isChecked()
            root['chunk_size'] = chunk_size
            root['overlap_large'] = overlap_large
            root['overlap_small'] = overlap_small
            self.accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MVSEP Music Separation Tool')
        self.setAcceptDrops(True)
        self.audio_model = AudioFileModel()
        self.setup_ui()
        self.load_theme()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        input_group = QGroupBox("Input Files")
        input_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        self.button_select_input_files = QPushButton("Select Input Files")
        self.button_select_input_files.setIcon(QIcon.fromTheme("document-open"))
        self.button_select_input_files.clicked.connect(self.dialog_select_input_files)
        self.button_remove_selected = QPushButton("Remove Selected")
        self.button_remove_selected.setIcon(QIcon.fromTheme("edit-delete"))
        self.button_remove_selected.clicked.connect(self.remove_selected_files)
        button_layout.addWidget(self.button_select_input_files)
        button_layout.addWidget(self.button_remove_selected)
        self.input_files_list = QListView()
        self.input_files_list.setModel(self.audio_model)
        self.input_files_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        input_layout.addLayout(button_layout)
        input_layout.addWidget(self.input_files_list)
        input_group.setLayout(input_layout)

        output_group = QGroupBox("Output Folder")
        output_layout = QHBoxLayout()
        self.output_folder_line_edit = QLineEdit()
        self.output_folder_line_edit.setReadOnly(True)
        self.button_select_output_folder = QPushButton("Select Folder")
        self.button_select_output_folder.setIcon(QIcon.fromTheme("folder"))
        self.button_select_output_folder.clicked.connect(self.dialog_select_output_folder)
        output_layout.addWidget(self.output_folder_line_edit)
        output_layout.addWidget(self.button_select_output_folder)
        output_group.setLayout(output_layout)

        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label = QLabel("Ready")
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        progress_group.setLayout(progress_layout)

        button_layout = QHBoxLayout()
        self.button_start = QPushButton('Start Separation')
        self.button_start.setIcon(QIcon.fromTheme("media-playback-start"))
        self.button_start.clicked.connect(self.execute_separation)
        self.button_stop = QPushButton('Stop')
        self.button_stop.setIcon(QIcon.fromTheme("media-playback-stop"))
        self.button_stop.clicked.connect(self.stop_separation)
        self.button_stop.setEnabled(False)
        self.button_settings = QPushButton('Settings')
        self.button_settings.setIcon(QIcon.fromTheme("preferences-system"))
        self.button_settings.clicked.connect(self.open_settings)
        button_layout.addWidget(self.button_start)
        button_layout.addWidget(self.button_stop)
        button_layout.addWidget(self.button_settings)

        layout.addWidget(input_group)
        layout.addWidget(output_group)
        layout.addWidget(progress_group)
        layout.addLayout(button_layout)

        self.statusBar().showMessage(f'MVSEP v{__VERSION__}')
        self.setup_menu()

    def setup_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        open_action = QAction('Open Files', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.dialog_select_input_files)
        file_menu.addAction(open_action)

        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        view_menu = menubar.addMenu('View')
        toggle_theme_action = QAction('Toggle Theme', self)
        toggle_theme_action.setShortcut('Ctrl+T')
        toggle_theme_action.triggered.connect(self.toggle_theme)
        view_menu.addAction(toggle_theme_action)

        help_menu = menubar.addMenu('Help')
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        QMessageBox.about(self, "About MVSEP",
                          f"MVSEP Music Separation Tool v{__VERSION__}\n\n"
                          "Enhanced by Fonk\n"
                          "Copyright © 2024")

    def load_theme(self):
        if root.get('dark_theme', False):
            self.set_dark_theme()
        else:
            self.set_light_theme()

    def toggle_theme(self):
        root['dark_theme'] = not root.get('dark_theme', False)
        self.load_theme()

    def set_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        self.setPalette(palette)

    def set_light_theme(self):
        self.setPalette(self.style().standardPalette())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.add_input_files(files)

    def add_input_files(self, files):
        valid_files = [f for f in files if os.path.isfile(f) and f.lower().endswith(('.wav', '.mp3', '.flac'))]
        if valid_files:
            self.audio_model.add_files(valid_files)
            root['input_files'] = self.audio_model.audio_files
            self.update_start_button_state()

    def remove_selected_files(self):
        indices = [index.row() for index in self.input_files_list.selectedIndexes()]
        self.audio_model.remove_files(indices)
        root['input_files'] = self.audio_model.audio_files
        self.update_start_button_state()

    def update_start_button_state(self):
        self.button_start.setEnabled(bool(root['input_files']) and bool(root['output_folder']))

    def execute_separation(self):
        if not root['input_files']:
            QMessageBox.warning(self, "Error", "No input files specified!")
            return
        if not root['output_folder']:
            QMessageBox.warning(self, "Error", "No output folder specified!")
            return

        self.button_start.setEnabled(False)
        self.button_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Processing...")

        options = {
            'input_audio': root['input_files'],
            'output_folder': root['output_folder'],
            'cpu': root['cpu'],
            'single_onnx': root['single_onnx'],
            'large_gpu': root['large_gpu'],
            'chunk_size': root['chunk_size'],
            'overlap_large': root['overlap_large'],
            'overlap_small': root['overlap_small'],
            'use_kim_model_1': root['use_kim_model_1'],
            'only_vocals': root['only_vocals'],
        }

        self.thread = QThread()
        self.worker = Worker(options)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.update_progress)
        self.worker.error.connect(self.show_error)
        self.thread.finished.connect(self.separation_finished)

        self.thread.start()

    def stop_separation(self):
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread'):
            self.thread.quit()
            self.thread.wait()
        self.separation_finished()

    def separation_finished(self):
        self.button_start.setEnabled(True)
        self.button_stop.setEnabled(False)
        self.progress_label.setText("Finished")
        QMessageBox.information(self, "Process Completed", "Audio separation completed successfully!")

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def show_error(self, error_message):
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.separation_finished()

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.load_settings()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            pass

    def dialog_select_input_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select input files",
            "",
            "Audio Files (*.wav *.mp3 *.flac);;All Files (*)"
        )
        if files:
            self.add_input_files(files)

    def dialog_select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if folder:
            root['output_folder'] = folder
            self.output_folder_line_edit.setText(folder)
            self.update_start_button_state()

def initialize_settings():
    root['input_files'] = []
    root['output_folder'] = ''
    root['cpu'] = False
    root['large_gpu'] = False
    root['single_onnx'] = False
    root['chunk_size'] = 500000
    root['overlap_large'] = 1
    root['overlap_small'] = 1
    root['use_kim_model_1'] = False
    root['only_vocals'] = False
    root['dark_theme'] = True

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_memory > 11.5:
            root['large_gpu'] = True
            root['single_onnx'] = False
        elif total_memory < 8:
            root['large_gpu'] = False
            root['single_onnx'] = True

def main():
    print(f'Version: {__VERSION__}')
    initialize_settings()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()