import tensorflow as tf
import multiprocessing as mp
import time


class RecongnitionResult:

    def __init__(self):
        self.data_ready = False
        self.info = "Not available"

    def run(self):
        pass


def EquationRecognizer(img, fetch_result):
    # TODO implement this
    # make sure we have a queue for inter process communication
    assert(isinstance(fetch_result, mp.queues.Queue))
    print("[INFO   ] received image with shape", img.shape)

    time.sleep(3)
    # fill the result from the computation
    result = RecongnitionResult()

    fetch_result.put(result)
    return
