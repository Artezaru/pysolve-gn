import numpy
import cv2
import os


def extend(
    image: numpy.ndarray,
    size: tuple[int, int] = (2, 4),
) -> numpy.ndarray:
    """
    Extend the input by creating a transparent grid of ``size[0]`` by ``size[1]`` with the same pixel size as the input image, and placing the input image at the center of this grid.
    The resulting image will have a transparent border around the original image.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be extended with shape (H, W, C), where H is the height, W is the width, and C is the number of channels (e.g., 3 for RGB or 4 for RGBA).

    size : tuple[int, int], optional
        A tuple specifying the size of the output image in terms of the number of pixels to add to the height and width of the input image.
        The default value is (2, 4), which means that 2 pixels will be added to the height and 4 pixels will be added to the width of the input image.


    Returns
    -------
    numpy.ndarray
        The extended image with the original image placed at the center and a transparent border around it. The shape of the output image will be (H * size[0], W * size[1], C).

    """
    if not isinstance(image, numpy.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if image.ndim != 3:
        raise ValueError("Input image must have three dimensions (H, W, C).")
    if not (
        isinstance(size, tuple)
        and len(size) == 2
        and all(isinstance(s, int) and s > 0 for s in size)
    ):
        raise ValueError("Size must be a tuple of two positive integers.")

    height, width, channels = image.shape
    extended_height = height * size[0]
    extended_width = width * size[1]
    extended_image = numpy.zeros(
        (extended_height, extended_width, channels), dtype=image.dtype
    )
    start_y = (extended_height - height) // 2
    start_x = (extended_width - width) // 2
    extended_image[start_y : start_y + height, start_x : start_x + width, :] = image
    return extended_image


def reduce(
    image: numpy.ndarray,
    size: tuple[int, int] = (2, 4),
) -> numpy.ndarray:
    """
    Reduce the input by cropping the central part of the image based on the specified size.
    The resulting image will be smaller than the original image, containing only the central portion defined by the size parameter.

    Parameters
    ----------
    image : numpy.ndarray
        The input image to be reduced with shape (H, W, C), where H is the height, W is the width, and C is the number of channels (e.g., 3 for RGB or 4 for RGBA).

    size : tuple[int, int], optional
        A tuple specifying the size of the output image in terms of the number of pixels to retain in the height and width of the input image.
        The default value is (2, 4), which means that only the central portion of the input image with a height of H//2 and a width of W//4 will be retained in the output image.


    Returns
    -------
    numpy.ndarray
        The reduced image containing only the central portion defined by the size parameter. The shape of the output image will be (H//size[0], W//size[1], C).

    """
    if not isinstance(image, numpy.ndarray):
        raise TypeError("Input image must be a numpy array.")
    if image.ndim != 3:
        raise ValueError("Input image must have three dimensions (H, W, C).")
    if not (
        isinstance(size, tuple)
        and len(size) == 2
        and all(isinstance(s, int) and s > 0 for s in size)
    ):
        raise ValueError("Size must be a tuple of two positive integers.")

    height, width, channels = image.shape
    reduced_height = height // size[0]
    reduced_width = width // size[1]
    start_y = (height - reduced_height) // 2
    start_x = (width - reduced_width) // 2
    reduced_image = image[
        start_y : start_y + reduced_height, start_x : start_x + reduced_width, :
    ]
    return reduced_image


# current_dir = os.path.dirname(os.path.abspath(__file__))
# icon_path = os.path.join(current_dir, "math.png")

# image = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)
# image = extend(image, size=(2, 4))
# cv2.imwrite(icon_path, image)
