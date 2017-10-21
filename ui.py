import numpy as np
import cv2
from cyclops import Scaler
from cyclops import Member

ui = np.zeros((540,300,3),np.uint8)
buttons = [((40,40), (260,100)), ((40,140), (260,200)), ((40,240), (260,300)), ((40,340), (260,400))]
color_area = [((50,460),(110,520)),((120,460),(180,520)),((190,460),(250,520))]
text = ['Get Scale','Add Member','Initialize','Start!']
ranges = [range(40,100), range(140,200), range(240,300), range(340,400)]
toffs = [(85,80),(65,180),(85,280),(110,380)]
font = cv2.FONT_HERSHEY_TRIPLEX
members = []

def buttonCallback(event, x, y, flags, param):
    global ui, nmembers
    if event == cv2.EVENT_LBUTTONUP:
        if x in range(40,260) and y in ranges[0]:
            sc = Scaler(0, 'trust.yaml')
            s = sc.run()
            print('{} px/meter'.format(s))
            button_color = (125, 0, 125) if s else (0, 255, 0)
            cv2.rectangle(ui, (buttons[0][0]), (buttons[0][1]), button_color, -1)
            cv2.putText(ui, text[0], toffs[0], font, 0.8, (0,0,0), 2)
        if x in range(40,260) and y in ranges[1]:
            if len(members) in range(0,3):
                m = Member()
                col = m.get_color()
                if col:
                    members.append(m)
                button_color = (125, 0, 125) if len(members) in range(1,4) else (0, 255, 0)
                cv2.rectangle(ui, (buttons[1][0]), (buttons[1][1]), button_color, -1)
                cv2.putText(ui, text[1], toffs[1], font, 0.8, (0,0,0), 2)
                for mb in range(0,len(members)):
                    c = members[mb].color
                    cv2.rectangle(ui, (color_area[mb][0]), (color_area[mb][1]), c, -1)
            else:
                print('Maximum 3 members can be added.')


def UI():
    global ui
    wname = 'Cyclops'
    cv2.namedWindow(wname,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(wname, buttonCallback)
    ui[:] = (100,100,100)
    ui[-100:] = (50, 50, 50)
    cv2.rectangle(ui, (buttons[0][0]), (buttons[0][1]), (0, 255, 0), -1)
    cv2.rectangle(ui, (buttons[1][0]), (buttons[1][1]), (0, 255, 0), -1)
    cv2.rectangle(ui, (buttons[2][0]), (buttons[2][1]), (0, 255, 0), -1)
    cv2.rectangle(ui, (buttons[3][0]), (buttons[3][1]), (0, 255, 0), -1)
    cv2.putText(ui, text[0], toffs[0], font, 0.8, (0,0,0), 2)
    cv2.putText(ui, text[1], toffs[1], font, 0.8, (0,0,0), 2)
    cv2.putText(ui, text[2], toffs[2], font, 0.8, (0,0,0), 2)
    cv2.putText(ui, text[3], toffs[3], font, 0.8, (0,0,0), 2)

    while True:
        cv2.imshow(wname, ui)
        cv2.waitKey(50)

UI()
