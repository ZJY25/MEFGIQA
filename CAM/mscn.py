import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_mscn_coefficients(image, window_size=(7, 7), sigma=7/6):
    # Convert the image to float
    image = image.astype(np.float64)

    # Calculate local mean
    local_mean = cv2.GaussianBlur(image, window_size, sigma)

    # Calculate local variance
    local_variance = cv2.GaussianBlur(image**2, window_size, sigma) - local_mean**2

    # Calculate MSCN coefficients
    mscn_coefficients = (image - local_mean) / (local_variance + 1)

    return mscn_coefficients

# Load an example image
image = cv2.imread('./t.png', cv2.IMREAD_GRAYSCALE)

# Calculate MSCN coefficients
mscn_coefficients = calculate_mscn_coefficients(image)

# Flatten the coefficients for histogram plotting
mscn_coefficients_flat = mscn_coefficients.flatten()

# Plot the histogram of MSCN coefficients with normalization
plt.hist(mscn_coefficients_flat, bins=200)
plt.show()
