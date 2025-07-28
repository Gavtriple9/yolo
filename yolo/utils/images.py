import PIL.Image as Image
import numpy as np
import cv2

from yolo.base import GridCell, Rectangle

BOUNDING_BOX_COLOR = (0, 255, 255)
BOUNDING_BOX_THICKNESS = 2

MIDPOINT_SIZE = 1
MIDPOINT_COLOR = (255, 0, 255)


def save_image_buf(image_buf: np.ndarray, path: str):
    """saves an image buffer to a file

    :param image_buf: image buffer as a numpy array
    :type image_buf: numpy.ndarray
    :param path: path to save the image to
    :type path: str

    :returns: None
    :rtype: None
    """

    image = Image.fromarray(image_buf, "RGB")
    image.save(path)


def get_rgba_int(r: int, g: int, b: int, a: int) -> int:
    return int(a % 256 << 24 | b % 256 << 16 | g % 256 << 8 | r % 256)


def draw_bbox_on_image(
    image_buf: np.ndarray, bbox_list: list[Rectangle], draw_midpoint: bool = True
) -> np.ndarray:
    """Draws bounding boxes on an image buffer.

    :param image_buf: image buffer as a numpy array
    :type image_buf: numpy.ndarray
    :param bbox_list: list of bounding boxes
    :type bbox_list: list[:class:`BoundingBox`]

    :returns: modified image buffer as a numpy array
    :rtype: numpy.ndarray
    """

    for bbox in bbox_list:
        center = bbox.center.integral()
        start_point = bbox.get_top_left_pixel()
        end_point = bbox.get_bottom_right_pixel()

        # Draw the bounding box
        cv2.rectangle(
            image_buf,
            start_point,
            end_point,
            BOUNDING_BOX_COLOR,
            BOUNDING_BOX_THICKNESS,
        )

        # Draw the center point
        if draw_midpoint:
            cv2.circle(
                image_buf,
                (center.x, center.y),
                MIDPOINT_SIZE,
                MIDPOINT_COLOR,
            )

    return image_buf


def generate_grid_cells(
    image_size: tuple[int, int], grid_size: tuple[int, int]
) -> np.ndarray[GridCell]:
    """Generates grid cells for an image.

    :param image_size: size of the image
    :type image_size: tuple (int, int)
    :param grid_size: size of the grid
    :type grid_size: tuple (int, int)

    :returns: list of grid cells
    :rtype: list[:class:`GridCell`]
    """
    grid_cells = np.empty(grid_size, dtype=GridCell)
    delta_w = image_size[0] // grid_size[0]
    delta_h = image_size[1] // grid_size[1]

    # Add extra pixels to last grid cells if
    # image size dimention is not divisible by grid size
    extra_w = image_size[0] % grid_size[0]
    extra_h = image_size[1] % grid_size[1]

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            w = delta_w
            h = delta_h
            if i == grid_size[0] - 1:
                w = delta_w + extra_w
            if j == grid_size[1] - 1:
                h = delta_h + extra_h
            cell = GridCell(w, h, i, j)
            grid_cells[i][j] = cell

    return grid_cells


def draw_grid_on_image(
    image_buf: np.ndarray, grid_cells: np.ndarray[GridCell]
) -> np.ndarray:
    """Draws grid cells on an image buffer.

    :param image_buf: image buffer as a numpy array
    :type image_buf: numpy.ndarray
    :param grid_cells: list of grid cells
    :type grid_cells: list[:class:`GridCell`]

    :returns: modified image buffer as a numpy array
    :rtype: numpy.ndarray
    """

    for i in range(grid_cells.shape[0]):
        for j in range(grid_cells.shape[1]):
            cell: GridCell = grid_cells[i][j]
            image_buf[:, i * cell.width] = [127, 127, 127]
            image_buf[j * cell.height, :] = [127, 127, 127]
            image_buf[:, i * cell.width] = [127, 127, 127]
            image_buf[j * cell.height, :] = [127, 127, 127]
            image_buf = draw_bbox_on_image(image_buf, cell.get_bbox_list())

    return image_buf
