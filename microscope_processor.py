import tifffile
import imageio.v2 as iio
import numpy as np
from scipy.signal import butter, freqz
from scipy.fft import fft2, ifft2, fftshift
from matplotlib import pyplot as plt    

class MicroscopeProcessor:
        
    def add_stack_img(self, original_stack_img):
        self.stack = original_stack_img
        self.C_stack, _, _ = self.stack.shape
        
    def add_single_img(self, original_single_img):
        self.img = original_single_img

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
    # Method to plot the spectrum from an image (private static method)
    @staticmethod
    def plot_spectrum(img, title, cmap='magma'):
        F = np.fft.fft2(img)
        F_shifted = np.fft.fftshift(F)
        magnitude = np.log1p(np.abs(F_shifted))
        magnitude = magnitude / np.max(magnitude)
        plt.imshow(magnitude, cmap=cmap)  # change colormap here
        plt.title(title)
        plt.axis('off')
    
    # Fourier-based demodulation method
    # Butterworth Low-Pass filter (private static method)
    @staticmethod
    def _butter_filter_lowpass(cuttoff_frequency, rows, cols, order):
        
        freqs_norm = np.linspace(-0.5, 0.5, rows, endpoint=False)
        
        # Ideal Butterworth filter
        
        Hy = 1.0 / (1.0 + (np.abs(freqs_norm) / cuttoff_frequency)**(2 * order))
        
        '''
        # butter + freqz filter
        b, a = butter(order, cuttoff_frequency, btype='low', analog=False)
        _, h = freqz(b, a, worN=rows, whole = True)
        Hy = np.abs(np.fft.fftshift(h))
        '''
        # Convert 1D to 2D
        Hy = Hy[:, np.newaxis]     # (rows, 1)
        Hx = np.ones((1, cols))    # (1, cols)
        
        return Hy @ Hx

    # Fourier-based demodulation method
    # Butterworth High-Pass filter (private static method)
    @staticmethod
    def _butter_filter_highpass(cuttoff_frequency, rows, cols, order):
        return 1 - MicroscopeProcessor._butter_filter_lowpass(cuttoff_frequency, rows, cols, order)

    # Fourier-based demodulation method
    # Apply filter in Fourier domain (private static method)
    @staticmethod
    def apply_filter(image_input, H):
        F = np.fft.fft2(image_input)
        F_shifted = np.fft.fftshift(F)
        F_filtered = F_shifted * H
        F_ifft = np.fft.ifft2(np.fft.ifftshift(F_filtered))
        img_filtered = np.real(F_ifft)
        return img_filtered, F_filtered

    # Fourier-based demodulation method
    def fourier_based_demodulation(self, T, order):
        
        rows, cols = self.img.shape
        cut_off_frequency = 1 / T    
        
        # 1) High-pass filtering
        H_high_filter = self._butter_filter_highpass(cut_off_frequency, rows, cols, order)
        high_filtered_img, _ = MicroscopeProcessor.apply_filter(self.img, H_high_filter)

        # 2) Frequency downshift via multiplication by sine/cosine references
        
        x = np.arange(rows)
        cos_mod = np.cos(2*np.pi*x / T)
        sin_mod = np.sin(2*np.pi*x / T)
            
        A_mix_img = high_filtered_img * cos_mod[:, None]
        B_mix_img = high_filtered_img * sin_mod[:, None]
        
        # 3) Low-pass filtering of the A and B signals
        # Retains only the frequency content of interest while discarding high-frequency artifacts
        H_low_filter = MicroscopeProcessor._butter_filter_lowpass(cut_off_frequency, rows, cols, order)
        A_low_filtered_img, _ = MicroscopeProcessor.apply_filter(A_mix_img, H_low_filter)
        B_low_filtered_img, _ = MicroscopeProcessor.apply_filter(B_mix_img, H_low_filter)
        
        # 4) Magnitude reconstruction
        # Combine the filtered A and B components:
        img_result = np.sqrt(A_low_filtered_img**2 + B_low_filtered_img**2)

        return high_filtered_img, A_mix_img, B_mix_img, A_low_filtered_img, B_low_filtered_img, img_result
    
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