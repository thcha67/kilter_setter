import sys
import os
sys.path.append(os.getcwd())

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import ScatterPlane
from kivy.uix.button import Button
from kivy.graphics import *

from scripts.kilter_utils import grade_translations, angle_translations, color_translations, get_all_holes_12x12, get_matrix_from_holes


class MyApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.grades = tuple(grade_translations.values())
        self.angles = tuple(angle_translations.values())

    def build(self):

        # Create the main layout
        layout = BoxLayout(orientation='vertical')

        # # Create the top bar with two side-by-side dropdowns
        top_bar1 = GridLayout(cols=2, size_hint=(1, None), height=80)

        label1 = Label(text='Angle:')
        label2 = Label(text='Grade:')
        spinner1 = Spinner(text=self.angles[0], values=self.angles)
        spinner2 = Spinner(text=self.grades[0], values=self.grades)

        top_bar1.add_widget(label1)
        top_bar1.add_widget(label2)
        top_bar1.add_widget(spinner1)
        top_bar1.add_widget(spinner2)

        top_bar2 = GridLayout(cols=3, size_hint=(1, None), height=40)

        button1 = Button(text='Generate')
        button2 = Button(text='Connect')
        button3 = Button(text='Save')

        top_bar2.add_widget(button1)
        top_bar2.add_widget(button2)
        top_bar2.add_widget(button3)

        # # Create the image below the top bar
        image = Image(source='assets/kilterboard_background.png', fit_mode="scale-down")
        image_with_points = ImageWithPoints()

        # Add widgets to the main layout
        layout.add_widget(top_bar1)
        layout.add_widget(top_bar2)
        layout.add_widget(image_with_points)

        return layout

class ImageWithPoints(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.all_holes = get_all_holes_12x12()

        with self.canvas:
            self.img = Image(source='assets/kilterboard_background.png')
            self.add_points(self.all_holes)
        
    def add_points(self, points):

        for x, y, color in points:
            scale_x
            Color(*hex_to_rgb(color_translations[color]))
            Ellipse(pos=(), size=(10, 10))

    def on_size(self, *args):
        self.img.size = self.size

    def on_pos(self, *args):
        self.img.pos = self.pos

def hex_to_rgb(hex_color):
    """
    Takes a hex color code and returns a tuple of RGB values from 0 to 1
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

if __name__ == '__main__':
    MyApp().run()
