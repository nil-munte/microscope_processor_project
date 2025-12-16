# microscope_gui.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QComboBox, QLabel, QSlider,
    QTabWidget, QSpinBox, QMessageBox
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

# Import your processor (must be in PYTHONPATH / same folder)
from microscope_processor import MicroscopeProcessor


def hide_imageview_ui(image_view: pg.ImageView):
    """Hide ImageView toolbar widgets for a cleaner UI."""
    try:
        image_view.ui.roiBtn.hide()
    except Exception:
        pass
    try:
        image_view.ui.menuBtn.hide()
    except Exception:
        pass
    try:
        image_view.ui.histogram.hide()
    except Exception:
        pass


class TifTab(QWidget):
    """Tab for TIF stack processing (3 projection algorithms)."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.stack_img = None
        # Create processor
        try:
            self.processor = MicroscopeProcessor()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot create MicroscopeProcessor:\n{e}")
            self.processor = None
            return            
        
        layout = QVBoxLayout(self)

        # Top controls: load / method / process
        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.load_btn = QPushButton("Load TIF Stack")
        self.load_btn.clicked.connect(self.load_tif)
        controls.addWidget(self.load_btn)

        controls.addWidget(QLabel("Method:"))
        self.method_box = QComboBox()
        self.method_box.addItems([
            "Average Projection",
            "Min-Max Projection",
            "Weighted Complex Average"
        ])
        controls.addWidget(self.method_box)

        self.process_btn = QPushButton("PROCESS")
        self.process_btn.clicked.connect(self.run_processing)
        controls.addWidget(self.process_btn)

        # Middle: original viewer and slider
        mid = QVBoxLayout()
        layout.addLayout(mid, stretch=1)

        mid.addWidget(QLabel("Original Stack (use slider to preview frames)"))
        self.original_view = pg.ImageView()
        hide_imageview_ui(self.original_view)
        mid.addWidget(self.original_view)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.update_frame)
        mid.addWidget(self.slider)

        # Bottom: processed result
        layout.addWidget(QLabel("Processed Result"))
        self.processed_view = pg.ImageView()
        hide_imageview_ui(self.processed_view)
        layout.addWidget(self.processed_view, stretch=1)

    def load_tif(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select TIF stack", "", "TIF files (*.tif *.tiff)")
        if not path:
            return
        try:
            self.stack_img = MicroscopeProcessor.load_tif(path)
            self.processor.add_stack_img(self.stack_img)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIF:\n{e}")
            return

        # display stack in ImageView - ImageView accepts 3D arrays (frames, height, width)
        # ensure array is numpy and in shape (frames, H, W)
        if isinstance(self.stack_img, np.ndarray) and self.stack_img.ndim == 3:
            self.original_view.setImage(self.stack_img[0].T)
            n_frames = self.stack_img.shape[0]
            self.slider.setMaximum(max(0, n_frames - 1))
            self.slider.setEnabled(True)
            self.slider.setValue(0)
            #self.update_frame
        else:
            QMessageBox.warning(self, "Unexpected data", "Loaded TIF has unexpected shape. Expecting 3D numpy array (C,H,W).")

    def update_frame(self):
        if self.stack_img is None:
            return
        idx = self.slider.value()
        # show single frame (2D) in the ImageView
        frame = self.stack_img[idx]
        self.original_view.setImage(frame.T)

    def run_processing(self):
        if self.stack_img is None:
            QMessageBox.warning(self, "No data", "Please load a TIF stack first.")
            return
            
        mode = self.method_box.currentText()
        try:
            if mode == "Average Projection":
                result = self.processor.average_projection()
            elif mode == "Min-Max Projection":
                result = self.processor.max_min_projection()
            elif mode == "Weighted Complex Average":
                result = self.processor.weighted_complex_average()
            else:
                QMessageBox.warning(self, "Unknown method", "Unknown processing method selected.")
                return
        except Exception as e:
            QMessageBox.critical(self, "Processing error", f"Processing failed:\n{e}")
            return

        if result is None:
            QMessageBox.critical(self, "Processing error", "Processing returned None. Check method names.")
            return

        # result should be 2D image
        self.processed_view.setImage(result.T)


class PngTab(QWidget):
    """Tab for PNG single-image Fourier demodulation (with intermediate selector)."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.single_img = None
        # Create processor
        try:
            self.processor = MicroscopeProcessor()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot create MicroscopeProcessor:\n{e}")
            self.processor = None
            return   

        # Store intermediates
        self._high_pass = None
        self._A_mix = None
        self._B_mix = None
        self._A_lp = None
        self._B_lp = None
        self._final = None

        layout = QVBoxLayout(self)

        # Top controls: load, T, order, process
        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.load_btn = QPushButton("Load PNG Image")
        self.load_btn.clicked.connect(self.load_png)
        controls.addWidget(self.load_btn)

        controls.addWidget(QLabel("Period T:"))
        self.period_spin = QSpinBox()
        self.period_spin.setRange(1, 1024)
        self.period_spin.setValue(16)
        controls.addWidget(self.period_spin)

        controls.addWidget(QLabel("Filter order:"))
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(3)
        controls.addWidget(self.order_spin)

        self.process_btn = QPushButton("PROCESS")
        self.process_btn.clicked.connect(self.run_processing)
        controls.addWidget(self.process_btn)

        # Middle: original + viewer selector for intermediates
        mid = QHBoxLayout()
        layout.addLayout(mid)

        left_col = QVBoxLayout()
        mid.addLayout(left_col)

        left_col.addWidget(QLabel("Original Image"))
        self.original_view = pg.ImageView()
        hide_imageview_ui(self.original_view)
        left_col.addWidget(self.original_view)

        # Right: single viewer and selector for intermediate images
        right_col = QVBoxLayout()
        mid.addLayout(right_col)

        right_col.addWidget(QLabel("View"))
        self.view_box = QComboBox()
        self.view_box.addItems([
            "Original",
            "High-pass Filtered",
            "A_mix",
            "B_mix",
            "A_low_pass",
            "B_low_pass",
            "Final Result"
        ])
        self.view_box.currentIndexChanged.connect(self.update_intermediate_view)
        right_col.addWidget(self.view_box)

        self.intermediate_view = pg.ImageView()
        hide_imageview_ui(self.intermediate_view)
        right_col.addWidget(self.intermediate_view)

    def load_png(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PNG image", "", "PNG files (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            self.single_img = MicroscopeProcessor.load_png(path)
            self.processor.add_single_img(self.single_img)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load PNG:\n{e}")
            return

        # show original
        self.original_view.setImage(self.single_img.T)

    def run_processing(self):
        if self.single_img is None:
            QMessageBox.warning(self, "No data", "Please load a PNG single image first.")
            return

        # attempt to create processor if not present
        if self.processor is None:
            try:
                self.processor = MicroscopeProcessor(None, self.single_img)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create MicroscopeProcessor:\n{e}")
                return

        T = int(self.period_spin.value())
        order = int(self.order_spin.value())
        
        try:
            # Expecting: high_pass_filtered_img, A_mix_img, B_mix_img, A_low_pass_img, B_low_pass_img, fourier_based_img
            (
                self._high_pass,
                self._A_mix,
                self._B_mix,
                self._A_lp,
                self._B_lp,
                self._final
            ) = self.processor.fourier_based_demodulation(T, order)
        except Exception as e:
            QMessageBox.critical(self, "Processing error", f"Fourier demodulation failed:\n{e}")
            return

        # Ensure original is available as well (for convenience)
        if self._final is None:
            QMessageBox.critical(self, "Processing error", "Demodulation returned None. Check implementation.")
            return

        # show selected view
        self.update_intermediate_view()

    def update_intermediate_view(self):
        key = self.view_box.currentText()
        arr = None
        if key == "Original":
            arr = self.single_img
        elif key == "High-pass Filtered":
            arr = self._high_pass
        elif key == "A_mix":
            arr = self._A_mix
        elif key == "B_mix":
            arr = self._B_mix
        elif key == "A_low_pass":
            arr = self._A_lp
        elif key == "B_low_pass":
            arr = self._B_lp
        elif key == "Final Result":
            arr = self._final

        if arr is None:
            # show blank or a notice: just clear view
            self.intermediate_view.clear()
            return

        self.intermediate_view.setImage(arr.T)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscope Processor GUI")
        self.resize(1200, 700)

        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        self.tif_tab = TifTab()
        self.png_tab = PngTab()

        tabs.addTab(self.tif_tab, "TIF Stack Processing")
        tabs.addTab(self.png_tab, "PNG Fourier Demodulation")

def main():
    app = QApplication(sys.argv)
    # optional pyqtgraph config: enable useOpenGL = True could speed display for big images
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()