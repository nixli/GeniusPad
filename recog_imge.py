import tensorflow as tf
import multiprocessing as mp
import time
from functools import reduce
from copy import deepcopy

class RecongnitionResult:

    def __init__(self):
        self.data_ready = False
        self.info = "Not available"

    def run(self):
        pass


def EquationRecognizer(img, pipe):
    # TODO implement this
    # make sure we have a queue for inter process communication
    assert(isinstance(pipe, mp.queues.Queue))
    print("[INFO   ] received image with shape", img.shape)

    DBSCAN(img)
    # fill the result from the computation
    #newimg = deepcopy(img)
    result = RecongnitionResult()

    pipe.put(result)
    return



def DBSCAN(drawing, eps = 1, ):
    row, col = drawing.shape

    for i in range(row//10 ):
        for j in range(col//10):
            if drawing[i*10][j*10] > 0:
                print(".", end='')
            else:
                print(" ", end='')
        print()

