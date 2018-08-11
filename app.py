#!/usr/bin/python3

import atexit
import multiprocessing as mp
import os
import signal

import genius_pad


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
