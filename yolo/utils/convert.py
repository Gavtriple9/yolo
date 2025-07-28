from yolo.base import Rectangle, Point


def objects_to_rects(objects: dict) -> list[(str, Rectangle)]:
    """Convert a list of annotations to a list of rectangles

    :param object: list of objects
    :type object: dict

    :returns: list of rectangles
    :rtype: list[Rectangle]
    """
    rects = []
    obj: dict
    for obj in objects:
        bbox = obj.get("bndbox", None)
        if bbox is None:
            continue
        rect = Rectangle.from_corners(
            Point(float(bbox["xmin"]), float(bbox["ymin"])),
            Point(float(bbox["ymax"]), float(bbox["ymax"])),
        )
        rect = rect * 0.25
        rects.append((obj["name"], rect))
    return rects
