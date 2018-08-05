from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.uix.label import Label
from kivy.graphics import Color, Line, Rectangle
from kivy.uix.floatlayout import  FloatLayout
from kivy.clock import Clock

import os
from random import random
from matplotlib.image import imread
import multiprocessing as mp
# local includes
from recog_imge import *

class ClipBoard(Widget):

    def on_touch_down(self, touch):
        color = (random(), 1, 1)
        with self.canvas:
            Color(*color, mode='hsv')
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=5)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]


class GeniusPad(App):

    def build(self):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        Window.clearcolor = (1, 1, 1, 1)
        self.parent = FloatLayout(size=(300, 300))
        self.painter = ClipBoard()

        clearbtn = Button(text='Clear', size_hint=(.1, .1),
                pos_hint={'x':.0, 'y':.0})

        self.savebtn = Button(text = "Compute", size_hint=(.1, .1),
                pos_hint={'x':.9, 'y':.0})
        clearbtn.bind(on_release=self.clear_canvas)
        self.savebtn.bind(on_release = self.init_compute)

        self.parent.add_widget(self.painter)
        self.parent.add_widget(clearbtn)
        self.parent.add_widget(self.savebtn)
        self.parent.bind(size=self._update_rect, pos=self._update_rect)

        self.img_count = 0
        self.result_fetched = mp.Queue()
        with self.parent.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.parent.size, pos=self.parent.pos)
        return self.parent

    def init_compute(self, _):
        img_array = self.generate_image()
        self.task = mp.Process(target=EquationRecognizer, args=(img_array, self.result_fetched))
        self.task.start()

        # fetch the result after one second
        Clock.schedule_once(self.fetch_compute_result, 1)

    def fetch_compute_result(self, dt):
        if self.task.is_alive():
            Clock.schedule_once(self.fetch_compute_result, .1)
        else:
            self.render_with_result()

    def render_with_result(self):
        # TODO: re draw based on the result given back
        print(self.result_fetched)

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def clear_canvas(self, obj):
        self.painter.canvas.clear()

    def generate_image(self):
        self.img_count += 1
        img_file = "/tmp/input_img"

        self.parent.export_to_png(img_file)
        with self.parent.canvas:
            anmi = Animation(opacity=0, duration=3)
            x,y = Window.size
            label_size = x/2, y/2
            msg = Label(text="Computing...", color = [0,0,0,1], size = label_size,
                        texture_size = label_size, bold = True)
            anmi.start(msg)

        image = imread(img_file)
        return image
