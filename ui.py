import numpy as np
import cv2
from cyclops import Scaler

ui = np.zeros((440,240,3),np.uint8)
buttons = [((40,40), (200,100)), ((40,140), (200,200)), ((40,240), (200,300)), ((40,340), (200,400))]
text = ["Get Scale","Add Agent","Initialize","Start!"]
ranges = [range(40,100), range(140,200), range(240,300), range(340,400)]
toffs = [(55,80),(50,180),(60,280),(85,380)]
font = cv2.FONT_HERSHEY_TRIPLEX

def buttonCallback(event, x, y, flags, param):
    global ui
    if event == cv2.EVENT_LBUTTONUP:
        if x in range(40,200) and y in ranges[0]:
            sc = Scaler(0, "trust.yaml")
            s = sc.run()
            print s,"px/meter"
            color = (125, 0, 125) if s else (0, 255, 0)
            cv2.rectangle(ui, (buttons[0][0]), (buttons[0][1]), color, -1)
            cv2.putText(ui, text[0], toffs[0], font, 0.8, (0,0,0), 2)

def UI():
    global ui
    wname = "Cyclops"
    # button1 = "Get Scale"
    # button2 = "Add Agent"
    # button3 = "Initialize"
    # button4 = "Start!"
    cv2.namedWindow(wname,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(wname, buttonCallback)
    ui[:] = (100,100,100)
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
