import tifffile
import numpy as np

class MicroscopeProcessor:

    def __init__(self, original_stack_img):
        self.stack = original_stack_img
        self.C, _, _ = self.stack.shape

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
        # k is np.arange(self.c): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # w_k has a final size of [C] Complex Values
        w_k = np.exp(1j * np.arange(self.C) * 2 * np.pi / self.C)
        # w_k[:, None, None]: reshapes w_k from C to C, 1, 1 so we can multiply weights and image at each corresponding channel
        return np.abs(np.sum(self.stack * w_k[:, None, None], axis = 0))
    
    # Method to load a TIFF into a numpy array
    # Input shape is (C, H, W), with C = 10
    @staticmethod
    def load_tif(tiff_file_path):
        return tifffile.imread(tiff_file_path)