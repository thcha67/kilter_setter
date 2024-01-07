import sys
sys.path.append(r'C:\Users\thoma\OneDrive\Documents\kilter\kilter_setter')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.button import Button

from scripts.kilter_utils import grade_translations, angle_translations


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

        # Add widgets to the main layout
        layout.add_widget(top_bar1)
        layout.add_widget(top_bar2)
        layout.add_widget(image)

        return layout

if __name__ == '__main__':
    MyApp().run()
