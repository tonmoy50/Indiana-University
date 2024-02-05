from scipy import fftpack
from scipy import ndimage
import imageio.v2 as imageio
import numpy as np
import os


def get_spectogram(image):
    fft2 = fftpack.fftshift(fftpack.fft2(image))
    return (np.log(abs(fft2)) * 255 / np.amax(np.log(abs(fft2)))).astype(np.uint8)


def get_kernel(size, sigma=1.0):
    x = np.zeros((size, size))
    x[size // 2, size // 2] = 1
    y = ndimage.filters.gaussian_filter(x, sigma=sigma)
    return y / y.sum()


if __name__ == "__main__":
    CURR_DIR = os.path.abspath(os.path.dirname(__file__))

    if not os.path.exists(os.path.join(CURR_DIR, "spectograms")):
        os.makedirs(os.path.join(CURR_DIR, "spectograms"))

    images = os.listdir(os.path.join(CURR_DIR, "images"))
    for img in images:
        image = imageio.imread(os.path.join(CURR_DIR, "images", img), mode="L")
        imageio.imsave(
            os.path.join(CURR_DIR, "spectograms", img),
            get_spectogram(image),
        )

    # load in an image, convert to grayscale if needed
    image = imageio.imread(os.path.join(CURR_DIR, "./pichu-secret-1.png"), mode="L")

    # take the fourier transform of the image
    fft2 = fftpack.fftshift(fftpack.fft2(image))

    # save FFT to a file. To help with visualization, we take
    # the log of the magnitudes, and then stretch them so they
    # fill the whole range of pixel values from 0 to 255.
    imageio.imsave(
        os.path.join(CURR_DIR, "fft.png"),
        (np.log(abs(fft2)) * 255 / np.amax(np.log(abs(fft2)))).astype(np.uint8),
    )

    # At this point, fft2 is just a numpy array and you can
    # modify it in order to modify the image in the frequency
    # space. Here's a little example (that makes a nearly
    # imperceptible change, but demonstrates what you can do.

    # The Secret Message is: HI
    # I tried to use low pass filtering with a gussian kernel to remove the secret message from the image. The gaussian kernel was applied through the entire image in a 2x2 window with an offset of 2.
    offset = size = 2
    kernel = get_kernel(size, 30)
    fft2_new = fft2.copy()

    for i in range(0, fft2_new.shape[0], offset):
        for j in range(0, fft2_new.shape[1], offset):
            if i + size > fft2_new.shape[0] or j + size > fft2_new.shape[1]:
                continue
            fft2_new[i : i + size, j : j + size] = (
                fft2_new[i : i + size, j : j + size] * kernel
            )
    fft2 = fft2_new

    # Workspace End

    # now take the inverse transform to convert back to an image
    ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))

    # and save the image

    imageio.imsave(
        os.path.join(CURR_DIR, "./fft-then-ifft.png"), ifft2.astype(np.uint8)
    )
