#!/usr/bin/python3

import genius_pad
import multiprocessing as mp
import atexit
import os
import signal

class Controller():
    def __init__(self):
        self.pad = genius_pad.GeniusPad()
        atexit.register(self.terminate_clean)

    def terminate_clean(self):
        pgrp = os.getpgrp()
        print("[INFO   ] Killing Process Group id:", pgrp)
        os.killpg(pgrp, signal.SIGTERM)
        print("[INFO   ] Application terminated")


if __name__ == '__main__':
    mp.set_start_method('fork')
    Controller().pad.run()
