from yolo.base import Rectangle, Point


def annos_to_rects(annos: dict) -> list[(str, Rectangle)]:
    """Convert a list of annotations to a list of rectangles

    :param annos: list of annotations
    :type annos: dict

    :returns: list of rectangles
    :rtype: list[Rectangle]
    """
    rects = []
    object: dict
    for object in annos.get("objects", []):
        bbox = object.get("bbox", None)
        rect = Rectangle.from_corners(
            Point(bbox["xmin"], bbox["ymin"]),
            Point(bbox["ymax"], bbox["ymax"]),
        )
        rects.append((object["name"], rect))
    return rects
