from microscope_processor import MicroscopeProcessor
from matplotlib import pyplot as plt
import os

# Input image
input_tif_image = "input_images/background_removal_raw.tif"

# Output image folder
output_folder = "output_images/"
os.makedirs(output_folder, exist_ok=True)

# Importing the original TIF image
original_stack_img = MicroscopeProcessor.load_tif(input_tif_image)
# Creating an instance of the MicroscopeProcessor class
processor = MicroscopeProcessor(original_stack_img)

# Frame combination algorithm modes

# Average projection
avg_projection_img = processor.average_projection()

# Min-Max projection
min_max_projection_img = processor.max_min_projection()

# Weighted complex average
weighted_complex_avg_img = processor.weighted_complex_average()

# Plot processed images
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(avg_projection_img, cmap='gray')
plt.title("Average Projection")

plt.subplot(1,3,2)
plt.imshow(min_max_projection_img, cmap='gray')
plt.title("Min-Max Projection")

plt.subplot(1,3,3)
plt.imshow(weighted_complex_avg_img, cmap='gray')
plt.title("Weighted Complex Average")

plt.show()

# Save processed images
plt.imsave(output_folder + "avg_projection_img.png", avg_projection_img, cmap='gray')
plt.imsave(output_folder+ "min_max_projection_img.png", min_max_projection_img, cmap='gray')
plt.imsave(output_folder + "weighted_complex_avg_img.png", weighted_complex_avg_img, cmap='gray')