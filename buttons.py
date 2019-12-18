import tkinter
import cv2, PySimpleGUI as sg
USE_CAMERA = 0      # change to 1 for front facing camera

layout = [sg.Image(filename='', key='image')
,sg.Button("Transfer", button_color=("white", "blue"), size=(6, 1))
]
print('d')
window, cap = sg.Window('Demo Application - OpenCV Integration', [layout, ], location=(0, 0), grab_anywhere=True), cv2.VideoCapture(USE_CAMERA)

win = window(timeout=20)[0]
print(win)
while win  is not None:
    win = window(timeout=20)[0]
    window['image'](data=cv2.imencode('.png', cap.read()[1])[1].tobytes())
    print(win)