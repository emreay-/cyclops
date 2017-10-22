import numpy as np
import cv2
from cyclops.utility import Scaler
from cyclops.utility import Member


class UserInterface(object):

    def __init__(self):
        self.ui = np.zeros((540,300,3),np.uint8)
        self.buttons = [((40,40), (260,100)), ((40,140), (260,200)), ((40,240), (260,300)), ((40,340), (260,400))]
        self.color_area = [((50,460),(110,520)),((120,460),(180,520)),((190,460),(250,520))]
        self.text = ['Get Scale','Add Member','Initialize','Start!']
        self.ranges = [range(40,100), range(140,200), range(240,300), range(340,400)]
        self.text_offset = [(85,80),(65,180),(85,280),(110,380)]
        self.font = cv2.FONT_HERSHEY_TRIPLEX
        self.members = []

    def buttonCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            if x in range(40,260) and y in self.ranges[0]:
                sc = Scaler(0, '/home/vetenskap/workspace/cyclops/param/trust.yaml')
                s = sc.run()
                print('{} px/meter'.format(s))
                button_color = (125, 0, 125) if s else (0, 255, 0)
                cv2.rectangle(self.ui, (self.buttons[0][0]), (self.buttons[0][1]), button_color, -1)
                cv2.putText(self.ui, self.text[0], self.text_offset[0], self.font, 0.8, (0,0,0), 2)

            if x in range(40,260) and y in self.ranges[1]:
                if len(self.members) in range(0,3):
                    m = Member()
                    col = m.get_color()
                    if col:
                        self.members.append(m)
                    button_color = (125, 0, 125) if len(self.members) in range(1,4) else (0, 255, 0)
                    cv2.rectangle(self.ui, (self.buttons[1][0]), (self.buttons[1][1]), button_color, -1)
                    cv2.putText(self.ui, self.text[1], self.text_offset[1], self.font, 0.8, (0,0,0), 2)
                    for mb in range(0,len(self.members)):
                        c = self.members[mb].color
                        cv2.rectangle(self.ui, (self.color_area[mb][0]), (self.color_area[mb][1]), c, -1)
                else:
                    print('Maximum 3 members can be added.')

    def run(self):
        window_name = 'Cyclops'
        cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.buttonCallback)
        self.ui[:] = (100,100,100)
        self.ui[-100:] = (50, 50, 50)
        cv2.rectangle(self.ui, (self.buttons[0][0]), (self.buttons[0][1]), (0, 255, 0), -1)
        cv2.rectangle(self.ui, (self.buttons[1][0]), (self.buttons[1][1]), (0, 255, 0), -1)
        cv2.rectangle(self.ui, (self.buttons[2][0]), (self.buttons[2][1]), (0, 255, 0), -1)
        cv2.rectangle(self.ui, (self.buttons[3][0]), (self.buttons[3][1]), (0, 255, 0), -1)
        cv2.putText(self.ui, self.text[0], self.text_offset[0], self.font, 0.8, (0,0,0), 2)
        cv2.putText(self.ui, self.text[1], self.text_offset[1], self.font, 0.8, (0,0,0), 2)
        cv2.putText(self.ui, self.text[2], self.text_offset[2], self.font, 0.8, (0,0,0), 2)
        cv2.putText(self.ui, self.text[3], self.text_offset[3], self.font, 0.8, (0,0,0), 2)

        while True:
            cv2.imshow(window_name, self.ui)
            cv2.waitKey(50)

if __name__=="__main__":
    interface = UserInterface()
    interface.run()
