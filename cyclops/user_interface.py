import os
import numpy as np
import cv2
from math import ceil
import logging

from cyclops.type_hints import *
from cyclops.utility import Scaler, Member
from cyclops.particle_filter import ParticleFilter


class Button(object):

    def __init__(self, top_left_xy: pixel_coord, bottom_right_xy: pixel_coord, 
                 idx: int, background_color: color_type = None, text: str = '', 
                 text_offset: pixel_coord = (20, 20), font: int = cv2.FONT_HERSHEY_TRIPLEX):
        self.top_left_xy = top_left_xy
        self.bottom_right_xy = bottom_right_xy
        self.idx = idx
        self.background_color = background_color
        self.text = text
        self.text_offset = text_offset
        self.text_position = ((self.top_left_xy[0] + self.text_offset[0]), 
                              (self.top_left_xy[1] + self.text_offset[1]))
        self.font = font
        self.x_range = range(self.top_left_xy[0], self.bottom_right_xy[0])
        self.y_range = range(self.top_left_xy[1], self.bottom_right_xy[1])
    
    def is_coordinates_in_button(self, x, y):
        return (x in self.x_range) and (y in self.y_range)


class Window(object):

    def __init__(self, name: str, number_of_rows_for_buttons: int, number_of_cols_for_buttons: int, 
                 button_height: int, button_width: int, padding_vertical: int, 
                 padding_horizontal: int, footer_height: int, 
                 background_color: color_type = (100, 100, 100), 
                 footer_background_color: color_type = (50, 50, 50), 
                 default_button_background_color: color_type = (0, 255, 0),
                 default_font: int = cv2.FONT_HERSHEY_TRIPLEX,
                 number_of_elements_in_footer: int = 3):

        self.name = name
        self.number_of_rows = number_of_rows_for_buttons
        self.number_of_cols = number_of_cols_for_buttons
        self.button_height = button_height
        self.button_width = button_width
        self.padding_vertical = padding_vertical
        self.padding_horizontal = padding_horizontal
        self.footer_height = footer_height
        self.background_color = background_color
        self.footer_background_color = footer_background_color
        self.default_button_background_color = default_button_background_color
        self.default_font = default_font
        self.number_of_elements_in_footer = number_of_elements_in_footer

        self.width = self.number_of_cols * (self.button_width + self.padding_horizontal) + self.padding_horizontal
        self.height = self.number_of_rows * (self.button_height + self.padding_vertical) + \
            self.padding_vertical + self.footer_height

        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.set_background_color()

        self.buttons = dict()
        self.set_footer_variables()
        self.footer_elements_counter = 0

    def set_footer_variables(self):
        _footer_width = self.width - 2 * self.padding_horizontal
        self.footer_element_size = self.footer_height - self.padding_vertical
        self.padding_footer = \
            round((_footer_width - self.number_of_elements_in_footer * self.footer_element_size) /
                  (self.number_of_elements_in_footer - 1))

    def set_background_color(self):
        self.canvas[:] = self.background_color
        self.canvas[-self.footer_height:] = self.footer_background_color

    def add_button(self, button_name: str, position_idx: int, 
                   text: str = '', text_offset: pixel_coord = (20, 20), 
                   background_color: color_type = None, font: int = None):
        background_color = self.default_button_background_color if background_color == None else background_color
        font = self.default_font if font == None else font

        top_left_xy, bottom_right_xy = self.get_button_bounding_box_from_position_idx(position_idx)
        self.buttons[button_name] = Button(top_left_xy, bottom_right_xy, position_idx, 
                                           background_color, text, text_offset, font)

    def get_button_bounding_box_from_position_idx(self, idx: int) -> Tuple[pixel_coord, pixel_coord]:
        row_number, col_number = self.get_row_and_col_number_from_position_idx(idx)
        return self.get_button_bounding_box_from_row_and_col_number(row_number, col_number)

    def get_row_and_col_number_from_position_idx(self, idx: int) -> Tuple[int, int]:
        row = ceil(float(idx / self.number_of_cols))
        col = idx % self.number_of_cols
        if col == 0:
            col = self.number_of_cols
        return row, col

    def get_button_bounding_box_from_row_and_col_number(
            self, row_number: int, col_number: int) -> Tuple[pixel_coord, pixel_coord]:
        top_left_x = col_number * self.padding_horizontal + (col_number - 1) * self.button_width
        top_left_y = row_number * self.padding_vertical + (row_number - 1) * self.button_height
        bottom_right_x = col_number * (self.padding_horizontal + self.button_width)
        bottom_right_y = row_number * (self.padding_vertical + self.button_height)
        return (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

    def update_buttons(self):
        for button in self.buttons.values():
            cv2.rectangle(self.canvas, button.top_left_xy, button.bottom_right_xy, button.background_color, -1)
            cv2.putText(self.canvas, button.text, button.text_position, button.font, 0.8, (0,0,0), 2)

    def match_pixel_coordinates_to_a_button(self, x, y):
        for button_name, button in self.buttons.items():
            if button.is_coordinates_in_button(x, y):
                return button_name
        return None

    def add_colored_box_in_footer(self, color: color_type):
        if self.footer_elements_counter != self.number_of_elements_in_footer:
            self.footer_elements_counter += 1
            _radius = self.footer_element_size // 2
            _x = self.padding_horizontal + \
                 ((2 * self.footer_elements_counter) - 1) * _radius + \
                 (self.footer_elements_counter - 1) * self.padding_footer
            _y = self.height - (self.footer_height // 2)

            cv2.circle(self.canvas, (_x, _y), _radius, color, thickness=-1)
            cv2.putText(self.canvas, 'M:{}'.format(self.footer_elements_counter), 
                        (_x - _radius, _y + _radius // 3), self.default_font, 0.8, (0,0,0), 2)


class UserInterface(object):

    def __init__(self):
        self.window = Window(name = 'Cyclops', number_of_rows_for_buttons = 4, number_of_cols_for_buttons = 1, 
                             button_height = 60, button_width = 220, padding_vertical = 40, 
                             padding_horizontal = 40, footer_height = 100)
        self.button_info = [('Get Scale', 1, (45, 40)) ,('Add Member', 2, (25, 40)), 
                            ('Initialize', 3, (45, 40)), ('Start!', 4, (70, 40))]
        self.button_color_when_not_yet_processed = (0, 255, 0)
        self.button_color_when_processed = (125, 0, 125)
        
        self.create_buttons()
        self.window.update_buttons()

        self.scale = None
        self.members = dict()
    
    def create_buttons(self):
        for button_name, position_idx, text_offset in self.button_info:
            self.window.add_button(button_name, position_idx, button_name, text_offset)

    def button_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            button_name = self.window.match_pixel_coordinates_to_a_button(x, y)
            
            if button_name == 'Get Scale':
                scaler = Scaler(0, '/home/vetenskap/workspace/cyclops/param/trust.yaml')
                self.scale = scaler.run()
                print('Scale: {} px/meter'.format(self.scale))
                if self.scale != None:
                    self.set_button_as_processed(button_name)

            if button_name == 'Add Member':
                if len(self.members) in range(0,3):
                    member = Member()
                    member.initialize_color()
                    if member.color:
                        self.members[member.id] = member
                        if len(self.members) in range(1, 4):
                            self.set_button_as_processed(button_name)
                            self.window.add_colored_box_in_footer(member.color)
                    else:
                        print('Member color is not initialized')
                else:
                    print('Maximum 3 members can be added.')
            
            if button_name == 'Initialize':
                if len(self.members) > 0:
                    for member in self.members.values():
                        member.initialize_location()
                        print(member.initial_location)

                    self.set_button_as_processed(button_name)
                else:
                    print('There are no members yet, add members first.')
            
            if button_name == 'Start!':
                if len(self.members) > 0:
                    parameters_file = os.path.join(os.getenv('CYCLOPS_PROJ_DIR'), 'param', 'filter_parameters.json')
                    camera_parameters_file = os.path.join(os.getenv('CYCLOPS_PROJ_DIR'), 'param', 'trust.yaml')
                    particle_filter = ParticleFilter(parameters_file=parameters_file, 
                                                     camera_parameters_file=camera_parameters_file,
                                                     camera_scale=self.scale, 
                                                     color_to_track=self.members[1].color)
                    particle_filter.initialize_particles(self.members[1].initial_location)
                    particle_filter.run()
                else:
                    print('There are no members yet, add members first.')

    def set_button_as_processed(self, button_name):
        self.window.buttons[button_name].background_color = self.button_color_when_processed
        self.window.update_buttons()

    def run(self):
        cv2.namedWindow(self.window.name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window.name, self.button_callback)

        while True:
            cv2.imshow(self.window.name, self.window.canvas)
            cv2.waitKey(50)


if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)
    interface = UserInterface()
    interface.run()
