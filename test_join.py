
from splitter.struct import *
from splitter.full_manager.full_video_processing import *
import cv2
import numpy as np
from splitter.constants import *

def create_repeating_file(name):
    cap = cv2.VideoCapture(name)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    #fourcc = cv2.VideoWriter_fourcc(*MP4V)
    write_vid = cv2.VideoWriter('cut3.mp4', fourcc, 1, (int(cap.get(3)),int(cap.get(4))))
    for i in range(20):
        test, frame = cap.read()
        if not test:
            break
        write_vid.write(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(20):
        test, frame = cap.read()
        if not test:
            break
        write_vid.write(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(20):
        test, frame = cap.read()
        if not test:
            break
        write_vid.write(frame)

def print_crops(crops, labels = None):
    for crop in crops:
        print('New Crop~~~~')
        print(crop['bb'].serialize())
        print(crop['label'])
        print(crop['all'])
    if labels:
        print('Labels~~~')
        print(labels)

def main():
    #Testing map
    # Test if cropping of one object works that slightly moves
    # Test if cropping of two objects that slightly move
    # But overlapps each other works
    data = []
    data.append([{'bb': Box(0, 0, 49, 49),  'label': 'test'}])
    data.append([{'bb': Box(1, 1, 50, 50),  'label': 'test'}])
    data.append([{'bb': Box(0, 0, 49, 49),  'label': 'test1'}])
    data.append([{'bb': Box(1, 1, 49, 49),  'label': 'test1'}])
    splitter = CropSplitter()
    map1 = splitter.map(data)
    map2 = splitter.map(data)
    # Test if empty labels join in tagger
    output = splitter.join(map1, map2)
    print_crops(output[0])
    # Test if same labels join in tagger
    # Test if join works with video_io

if __name__ == '__main__':
    create_repeating_file('cut2.mp4')