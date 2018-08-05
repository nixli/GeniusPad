#!/usr/bin/python3

import genius_pad
import multiprocessing as mp

class Controller():
    def __init__(self):
        self.pad = genius_pad.GeniusPad()



if __name__ == '__main__':
    mp.set_start_method('fork')
    Controller().pad.run()
