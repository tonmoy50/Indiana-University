from scipy import fftpack
from scipy import stats as st
from scipy import ndimage
import imageio.v2 as imageio
import numpy as np
import os


def get_spectogram(image):
    fft2 = fftpack.fftshift(fftpack.fft2(image))
    return (np.log(abs(fft2)) * 255 / np.amax(np.log(abs(fft2)))).astype(np.uint8)


def gkern(kernlen=21, nsig=3):
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


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

    # fft2[1, 1] = fft2[1, 1] + 1

    # Workspace Start
    fft2 = ndimage.gaussian_filter(fft2, 3, radius=30)

    imageio.imsave(
        os.path.join(CURR_DIR, "convolved.png"),
        (np.log(abs(fft2)) * 255 / np.amax(np.log(abs(fft2)))).astype(np.uint8),
    )

    # Workspace End

    # now take the inverse transform to convert back to an image
    ifft2 = abs(fftpack.ifft2(fftpack.ifftshift(fft2)))

    # and save the image

    imageio.imsave(
        os.path.join(CURR_DIR, "./fft-then-ifft.png"), ifft2.astype(np.uint8)
    )
