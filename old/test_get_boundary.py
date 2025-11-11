from skimage.segmentation import find_boundaries
import numpy as np

if __name__ == "__main__":
    H, W = 10, 10
    mask = np.zeros((H, W), dtype=bool)
    mask[2:8, 2:8] = True
    boundary = find_boundaries(mask, mode='inner', background=False)  # [H, W]

    # plot the mask and its boundary
    import matplotlib.pyplot as plt

    plt.subplot(1, 3, 1)
    plt.title("Mask")
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Boundary")
    plt.imshow(boundary, cmap='gray')
    
    # mask without boundary
    plt.subplot(1, 3, 3)
    plt.title("Mask without Boundary")
    plt.imshow(mask & ~boundary, cmap='gray')
    plt.show()

