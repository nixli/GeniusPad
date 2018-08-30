import os
import queue
from random import random, randint
from kivy.graphics.instructions import InstructionGroup
from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import (
    Translate, Fbo, ClearColor, ClearBuffers, Scale, Color, Line, Rectangle)
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
import tensorflow as tf
# local includes
from recog_imge import *


class ClipBoard(Widget):
    def on_touch_down(self, touch):
        color = (random(), 1, 1)
        with self.canvas:
            Color(*color, mode='hsv')
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=2)

    def on_touch_move(self, touch):
        touch.ud['line'].points += [touch.x, touch.y]

    def make_fbo(self, *args):

        if self.parent is not None:
            canvas_parent_index = self.parent.canvas.indexof(self.canvas)
            if canvas_parent_index > -1:
                self.parent.canvas.remove(self.canvas)

        fbo = Fbo(size=self.size, with_stencilbuffer=True)

        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            Scale(1, -1, 1)
            Translate(-self.x, -self.y - self.height, 0)

        fbo.add(self.canvas)
        fbo.draw()
        fbo.remove(self.canvas)

        if self.parent is not None and canvas_parent_index > -1:
            self.parent.canvas.insert(canvas_parent_index, self.canvas)

        return fbo


class GeniusPad(App):
    def build(self):

        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        Window.clearcolor = (1, 1, 1, 1)
        self.parent = FloatLayout(size=(300, 300))
        self.painter = ClipBoard()

        clearbtn = Button(text='Clear', size_hint=(.1, .1),
                          pos_hint={'x': .0, 'y': .0})

        self.savebtn = Button(text="Compute", size_hint=(.1, .1),
                              pos_hint={'x': .9, 'y': .0})
        clearbtn.bind(on_release=self.clear_canvas)
        self.savebtn.bind(on_release=self.init_compute)

        self.parent.add_widget(self.painter)
        self.parent.add_widget(clearbtn)
        self.parent.add_widget(self.savebtn)
        self.parent.bind(size=self._update_rect, pos=self._update_rect)

        self.img_count = 0
        self.pipe = mp.Queue()
        self.cluster_boxes = None
        with self.parent.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.parent.size, pos=self.parent.pos)
        return self.parent

    def init_compute(self, _):
        self.computation_start = time.time()
        # remove bounding boxes
        if self.cluster_boxes is not None:
            self.painter.canvas.remove(self.cluster_boxes)

        img_array = self.generate_image_data()

        self.task = mp.Process(target=EquationRecognizer,
                               args=(img_array, self.pipe))
        self.task.start()

        # fetch the result after one second
        Clock.schedule_once(self.fetch_compute_result, .1)

    def fetch_compute_result(self, dt):
        if self.task.is_alive():
            Clock.schedule_once(self.fetch_compute_result, .1)
        else:
            self.render_with_result()

    def render_with_result(self):
        try:
            result = self.pipe.get(timeout=1)
        except queue.Empty:
            pr_info("Something went wrong with the computation", mode="W")
            return
        pr_info("Computation Result:", result.info)

        self.cluster_boxes = InstructionGroup()
        self.cluster_boxes.add(Color(0, 0, 0, 1))
        for info in result.data:
            pr_info("Rectangle: ", info[0], info[1], info[2], info[3])
            self.cluster_boxes.add(
                Line(points=(info[3], self.painter.height - info[0],
                             info[2], self.painter.height - info[0],
                             info[2], self.painter.height - info[1],
                             info[3], self.painter.height - info[1],
                             info[3], self.painter.height - info[0])))

        self.painter.canvas.add(self.cluster_boxes)
        pr_info("Computation took {:.3f}s".format(time.time() - self.computation_start) )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def clear_canvas(self, obj):
        self.painter.canvas.clear()

    def generate_image_data(self):

        fbo = self.painter.make_fbo()
        with self.parent.canvas:
            anmi = Animation(opacity=0, duration=3)
            x, y = Window.size
            label_size = x / 2, y / 2
            msg = Label(text="Computing...", color=[0, 0, 0, 1], size=label_size,
                        texture_size=label_size, bold=True)
            anmi.start(msg)

        # each pixel is 8 bits, but the idea of color needs to be abstracted out
        image_array = np.frombuffer(fbo.pixels, dtype=np.uint32)
        has_drawing = image_array > 0

        # get rid of the unnecessary value, clone the only knowledge we need
        processed_array = np.zeros(shape=image_array.shape, dtype=np.uint8)
        processed_array[has_drawing] = 1

        # recover the shape, took me one hour to find this out
        processed_array.shape = (fbo.size[1], fbo.size[0])

        #np.save("./plus"+str(randint(0,99999)), processed_array)
        return processed_array
