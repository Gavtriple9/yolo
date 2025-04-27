import PIL.Image as Image
import numpy as np
import cv2

from yolo.gridcell import GridCell
from yolo.bbox import BoundingBox

def save_image_buf(image_buf, path, size=(448, 448, 3)):
    """saves an image buffer to a file
    
    :param image_buf: image buffer as a numpy array
    :type image_buf: numpy.ndarray
    :param path: path to save the image to
    :type path: str
    :param size: size of the image, defaults to (448, 448, 3)
    :type size: tuple, optional
    
    :returns: None
    :rtype: None
    """
    
    image = Image.new("RGBA", size)
    image.putdata(image_buf)
    image.save(path)

def get_rgba_int(r, g, b, a):
    return int(a % 256 << 24 | b % 256 << 16 | g % 256 << 8 | r % 256)

    
def draw_bbox_on_image(image_buf: np.ndarray, bbox_list: list[BoundingBox]) -> np.ndarray:
    """Draws bounding boxes on an image buffer.
    
    :param image_buf: image buffer as a numpy array
    :type image_buf: numpy.ndarray
    :param bbox_list: list of bounding boxes
    :type bbox_list: list[:class:`BoundingBox`]
    
    :returns: modified image buffer as a numpy array
    :rtype: numpy.ndarray
    """
    for bbox in bbox_list:
        start_point = bbox.get_top_left()
        end_point = bbox.get_bottom_right()
        color = (0, 255, 255)
        thickness = 2
        point_x = int(bbox.x)
        point_y = int(bbox.y)
        point_size = 1
        point_color = (255, 0, 255)
        
        cv2.rectangle(image_buf, start_point, end_point, color, thickness)
        cv2.circle(image_buf, (point_x, point_y), point_size, point_color, -1)
        
    return image_buf

    
def generate_grid_cells(image_size: tuple[int, int], grid_size: tuple[int, int]) -> np.ndarray[GridCell]:
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

    
def draw_grid_on_image(image_buf: np.ndarray, grid_cells: np.ndarray[GridCell]) -> np.ndarray:
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
