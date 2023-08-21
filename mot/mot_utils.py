
import numpy as np


def tlbr(self):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = self.tlwh.copy()
    ret[2:] += ret[:2]
    return ret


def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret


def tlbr_to_tlwh(tlbr):
    """Converts top-left bottom-right format to top-left width height format."""
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret


def tlwh_to_tlbr(tlwh):
    """Converts tlwh bounding box format to tlbr format."""
    ret = np.asarray(tlwh).copy()
    ret[2:] += ret[:2]
    return ret


def xywh_to_tlbr(xywh):
    """Converts xywh bounding box format to tlbr format."""
    ret = np.zeros((4, ))
    ret[0:2] = xywh[0:2] - xywh[2:4]/2
    ret[2:4] = xywh[0:2] + xywh[2:4]/2
    return ret


# convert test
if __name__ == "__main__":

    a = np.array([[5., 5., 3., 3., 1.], [5., 5., 3., 3., 1.], [5., 5., 3., 3., 1.]])
    c = np.asarray(a[:, 0:4]).copy()
    c[:, 0:2] = a[:, 0:2] - a[:, 2:4]/2
    c[:, 2:4] = a[:, 0:2] + a[:, 2:4]/2

    d = np.concatenate((a[:, 0:2] - a[:, 2:4]/2, a[:, 0:2] + a[:, 2:4]/2), axis=1)

    print("c: ", c)
    print("d: ", d)

    # addTen = np.vectorize(xywh_to_tlbr)
    # print(a[:, :4])
    # b = addTen(a[:, :4])
    # print(b)
