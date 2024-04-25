#! "D:\python projects\project 2\myenv\Scripts\python.exe"import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define Gaussian PSF
def gaussian_psf(size, sigma):
    """Generate a 2D Gaussian Point Spread Function (PSF)"""
    kernel_size = size // 2
    x = torch.linspace(-kernel_size, kernel_size, size)
    y = torch.linspace(-kernel_size, kernel_size, size)
    x, y = torch.meshgrid(x, y)
    psf = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    psf /= psf.sum()
    return psf.float()

# Convolution function
def convolve(image, kernel):
    """Convolve an image with a kernel"""
    # Ensure image is 4D tensor: [batch_size, channels, height, width]
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    # Add channel dimension to kernel
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    
    # Perform convolution
    convolved = F.conv2d(image, kernel, padding=kernel.shape[-1] // 2)
    
    return convolved.squeeze()

# Generate a sample image
image = torch.zeros(1, 1, 64, 64)  # single-channel 64x64 image
image[:, :, 28:36, 28:36] = 1.0  # square patch in the center

# Generate the PSF
psf_size = 15
sigma = 2.0
psf = gaussian_psf(psf_size, sigma)

# Convolve the image with the PSF
blurred_image = convolve(image, psf)

# Plot the results
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(image.squeeze().numpy(), cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(psf.numpy(), cmap='gray')
axs[1].set_title('Point Spread Function (PSF)')
axs[1].axis('off')

axs[2].imshow(blurred_image.numpy(), cmap='gray')
axs[2].set_title('Blurred Image')
axs[2].axis('off')

plt.show()
