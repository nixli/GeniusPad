import tensorflow as tf
import numpy as np
import multiprocessing as mp
import time

from functools import reduce
from copy import deepcopy


def pr_info(*args, mode="I"):
    color_modes = {
        "I": ("INFO"   , "\x1b[1;32m", "\x1b[0m"),
        "W": ("WARNING", "\x1b[1;33m", "\x1b[0m"),
        "E": ("ERROR"  , "\x1b[1;31m", "\x1b[0m")
    }

    print("[{}{:<7}{}] [{:<12}] {}".format(color_modes[mode][1], color_modes[mode][0], color_modes[mode][2],
                                            "APPLICATION", " ".join([str(i) for i in args])))


def debug_img(drawing):
    row, col = drawing.shape

    for i in range(row//10 ):
        for j in range(col//10):
            if drawing[i*10][j*10] > 0:
                print(".", end='')
            else:
                print(" ", end='')
        print()


class RecognitionResult:

    def __init__(self):
        self.data_ready = False
        self.info = "Not available"

    def run(self):
        pass


class Point:

    def __init__(self, x, y, clusterid = None):
        self.x = x
        self.y = y
        self.clusterid = clusterid


def EquationRecognizer(img, pipe):
    # TODO implement this
    # make sure we have a queue for inter process communication
    assert(isinstance(pipe, mp.queues.Queue))
    pr_info("received image with shape", img.shape)

    DBSCAN(img)
    # fill the result from the computation
    result = RecognitionResult()

    pipe.put(result)
    return


def DBSCAN(drawing, eps=1, minpts=5):
    pts = []
    # get points and their pixel location
    pt_iter = np.nditer(drawing, flags=['multi_index', ])
    while not pt_iter.finished:
        if pt_iter[0] == 1:
            pts.append(Point(*pt_iter.multi_index))
        pt_iter.iternext()
    pr_info("Number of Points: ", len(pts))

#http://www.cse.buffalo.edu/faculty/azhang/cse601/density-based.ppt
#    for point in pts:
