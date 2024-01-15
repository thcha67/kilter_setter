import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import cv2
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.graphics import *
import matplotlib.pyplot as plt

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
        holes = get_all_holes_12x12()
        create_image(holes)
        points_image = Image(source='src/android_app/assets/kilterboard_background_all_holes.png', fit_mode="scale-down")
        

        # Add widgets to the main layout
        layout.add_widget(top_bar1)
        layout.add_widget(top_bar2)
        layout.add_widget(points_image)

        return layout

def create_image(holes):
    """
    Creates the image of the KilterBoard with the holds
    """
    fig = plt.figure(figsize=(10.8, 12.22))
    plt.imshow(plt.imread('src/android_app/assets/kilterboard_background.png'), extent=[0, 1080, 0, 1222])
    plt.axis('off')

    for x, y, color in holes:
        x = x * 7.491
        y = y * 7.509 + 20
        plt.scatter(x, y, s=1100, edgecolors=color_translations[color], facecolors='none', linewidths=2)

    ax = plt.axes()
    ax.set_axis_off()
    ax.set_facecolor('none')
    fig.savefig('src/android_app/assets/kilterboard_background_all_holes.png', format='png', dpi=130, transparent=True, bbox_inches='tight', pad_inches=0)



def hex_to_rgb(hex_color):
    """
    Takes a hex color code and returns a tuple of RGB values from 0 to 1
    """
    hex_color = hex_color.lstrip('#')
    return tuple([int(hex_color[i:i+2], 16) for i in (0, 2, 4)] + [255])

if __name__ == '__main__':
    MyApp().run()
    pass
