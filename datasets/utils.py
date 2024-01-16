import numpy as np


def label2yolobox(labels, info_img, maxsize, lrflip=False):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    h, w, nh, nw, dx, dy, _ = info_img
    x1 = labels[:, 0] / w
    y1 = labels[:, 1] / h
    x2 = (labels[:, 0] + labels[:, 2]) / w
    y2 = (labels[:, 1] + labels[:, 3]) / h
    labels[:, 0] = (((x1 + x2) / 2) * nw + dx) / maxsize
    labels[:, 1] = (((y1 + y2) / 2) * nh + dy) / maxsize

    labels[:, 2] = (labels[:, 2] * (nw / w / maxsize))
    labels[:, 3] = (labels[:, 3] * (nh / h / maxsize))

    # labels[:, 2] *= nw / w / maxsize
    # labels[:, 3] *= nh / h / maxsize
    labels[:, :4] = np.clip(labels[:, :4], 0., 0.99)
    if lrflip:
        labels[:, 0] = 1 - labels[:, 0]
    return labels
