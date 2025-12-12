import tifffile
import imageio.v2 as iio
import numpy as np
from scipy.signal import butter, filtfilt

class MicroscopeProcessor:

    def __init__(self, original_stack_img, original_single_img):
        self.stack = original_stack_img
        self.img = original_single_img
        self.C_stack, _, _ = self.stack.shape

    # Frame combination algorithm. Average projection: compute the mean across all C frames
    # Computing the sum across axis 0 (C dimension): I_result = Σ_{i=0}^{C-1} I_i(x, y)
    def average_projection(self):
        return np.sum(self.stack, axis = 0)
    
    # Frame combination algorithm. Max-min projection: compute the difference between the maximum and minimum intensity projections.
    # Computing the difference accross axis 0 (C dimension): I_result = MAX(I_i(x, y)) - MIN(I_i(x, y))
    # MAX/MIN: for each pixel, gets the one of maximum/minimum intensity across all C frames
    def max_min_projection(self):
        return np.max(self.stack, axis=0) - np.min(self.stack, axis=0)
    
    # Frame combination algorithm. Weighted complex average: compute a weighted sum using complex exponential weights 𝑤𝑘 = exp 𝑖 𝑘 𝜋/𝐶 , and output the magnitude of the result. At the end, take the absolute value as result.
    def weighted_complex_average(self):
        # wk = exp(i · k · 2π / C)
        # k is np.arange(self.C_stack): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # w_k has a final size of [C] Complex Values
        w_k = np.exp(1j * np.arange(self.C_stack) * 2 * np.pi / self.C_stack)
        # w_k[:, None, None]: reshapes w_k from C to C, 1, 1 so we can multiply weights and image at each corresponding channel
        return np.abs(np.sum(self.stack * w_k[:, None, None], axis = 0))
    
    # Fourier-based demodulation method
    # Butterworth filter (2D), indicating type ('high', 'low' for this project) (private method)
    # The order of a Butterworth filter controls how steeply the filter transitions from passband to stopband
    def _butter_filter(self, cuttoff_frequency, order, type):
        # The cutoff frequencies in this method are normalized by the Nyquist frequency
        Wn = cuttoff_frequency/0.5
        b, a = butter(order, Wn, btype=type, analog=False)
        return b, a
    
    # Fourier-based demodulation method
    # Butterworth filter application to the image (private method)
    def _apply_filter(self, img, b, a):
        # Apply the 2-D Vutterworth filter along both axes using filtfilt for zero-phase filtering
        # axis = 0 -> H
        # axis = 1 -> W
        img_filt_x = filtfilt(b, a, img, axis=0)
        img_filt_xy = filtfilt(b, a, img_filt_x, axis=1)
        
        return img_filt_xy
    
    # Fourier-based demodulation method
    def fourier_based_demodulation(self, T, order):
        
        cut_off_frequency = 1 / T
        
        # 1) High-pass filtering
        bh, ah = self._butter_filter(cut_off_frequency, order, 'highpass')
        high_pass_filtered_img = self._apply_filter(self.img, bh, ah)
        
        # 2) Frequency downshift via multiplication by sine/cosine references
        # Modulation along width
        x = np.arange(self.img.shape[0])[:, None]
        # Cosine modulation of period T
        cos_mod = np.cos(2*np.pi*x / T)
        # Sine modulation of period T
        sin_mod = np.sin(2*np.pi*x / T)
        
        A_mix_img = high_pass_filtered_img * cos_mod
        B_mix_img = high_pass_filtered_img * sin_mod
        
        # 3) Low-pass filtering of the A and B signals
        # Retains only the frequency content of interest while discarding high-frequency artifacts
        bl, al = self._butter_filter(cut_off_frequency, order, 'lowpass')
        A_low_pass_img = self._apply_filter(A_mix_img, bl, al)
        B_low_pass_img = self._apply_filter(B_mix_img, bl, al)
        
        # 4) Magnitude reconstruction
        # Combine the filtered A and B components:
        img_result = np.sqrt(A_low_pass_img**2 + B_low_pass_img**2)

        return high_pass_filtered_img, A_mix_img, B_mix_img, A_low_pass_img, B_low_pass_img, img_result
    
    # Method to load a TIFF into a numpy array
    # Input shape is (C, H, W), with C = 10
    @staticmethod
    def load_tif(tiff_file_path):
        return tifffile.imread(tiff_file_path)
    
    # Method to load a PNG image into a numpy array
    # Input shape is (H, W)
    @staticmethod
    def load_png(png_file_path):
        return iio.imread(png_file_path)