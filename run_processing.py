from microscope_processor import MicroscopeProcessor
from matplotlib import pyplot as plt
import os

# Input images
input_tif_image = "input_images/background_removal_raw.tif"
input_png_image = "input_images/a.png"

# Output image folder
output_folder = "output_images/"
os.makedirs(output_folder, exist_ok=True)

# Importing the original TIF image
original_stack_img = MicroscopeProcessor.load_tif(input_tif_image)
original_single_img = MicroscopeProcessor.load_png(input_png_image)

# Creating an instance of the MicroscopeProcessor class
processor = MicroscopeProcessor(original_stack_img, original_single_img)

# ------------ Frame combination algorithm 

# Average projection
avg_projection_img = processor.average_projection()

# Min-Max projection
min_max_projection_img = processor.max_min_projection()

# Weighted complex average
weighted_complex_avg_img = processor.weighted_complex_average()

# ------------ Fourier-based demodulation

# Period (in pixels)
T = 16
# Order: between 3-4 to obtain a good balance between removing background and preserving image details
order = 3

high_pass_filtered_img, A_mix_img, B_mix_img, A_low_pass_img, B_low_pass_img, fourier_based_img = processor.fourier_based_demodulation(T, order)

# ------------ Plot processed images

plt.figure(1, figsize=(12,4))

# Frame combination algorithm
plt.subplot(1,3,1)
plt.imshow(avg_projection_img, cmap='gray')
plt.title("Average Projection")

plt.subplot(1,3,2)
plt.imshow(min_max_projection_img, cmap='gray')
plt.title("Min-Max Projection")

plt.subplot(1,3,3)
plt.imshow(weighted_complex_avg_img, cmap='gray')
plt.title("Weighted Complex Average")

# Fourier-based demodulation

plt.figure(2, figsize=(12,4))

plt.subplot(2,3,1)
plt.imshow(high_pass_filtered_img, cmap='gray')
plt.title("High-Pass Filtered Image")

plt.subplot(2,3,2)
plt.imshow(A_mix_img, cmap='gray')
plt.title("A_mix Image")

plt.subplot(2,3,3)
plt.imshow(B_mix_img, cmap='gray')
plt.title("B_mix Image")

plt.subplot(2,3,4)
plt.imshow(A_low_pass_img, cmap='gray')
plt.title("Low-Pass Filtered A_mix Image")

plt.subplot(2,3,5)
plt.imshow(B_low_pass_img, cmap='gray')
plt.title("Low-Pass Filtered B_mix Image")

plt.subplot(2,3,6)
plt.imshow(fourier_based_img, cmap='gray')
plt.title("Fourier-Based Demodulation Image")

plt.show()

# ------------ Save processed images
plt.imsave(output_folder + "avg_projection_img.png", avg_projection_img, cmap='gray')
plt.imsave(output_folder+ "min_max_projection_img.png", min_max_projection_img, cmap='gray')
plt.imsave(output_folder + "weighted_complex_avg_img.png", weighted_complex_avg_img, cmap='gray')
plt.imsave(output_folder + "high_pass_filtered_img.png", high_pass_filtered_img, cmap='gray')
plt.imsave(output_folder+ "A_mix_img.png", A_mix_img, cmap='gray')
plt.imsave(output_folder + "B_mix_img.png", B_mix_img, cmap='gray')
plt.imsave(output_folder + "A_low_pass_img.png", A_low_pass_img, cmap='gray')
plt.imsave(output_folder+ "B_low_pass_img.png", B_low_pass_img, cmap='gray')
plt.imsave(output_folder + "fourier_based_img.png", fourier_based_img, cmap='gray')