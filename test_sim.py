import sys
from experiments.environ import *

from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.tracking.background import FixedCameraBGFGSegmenter
from splitter.optimizer.splitter import SplitterOptimizer

from splitter.struct import *
from splitter.dataflow.map import *
from splitter.full_manager.full_manager import *
from splitter.utils.testing_utils import *
from splitter.dataflow.agg import *
from splitter.tracking.contour import *
from splitter.tracking.event import *
from splitter.core import *
from splitter.simple_manager.manager import *
import os

import cv2
import numpy as np
from scipy import stats


from splitter.dataflow.xform import *
from splitter.utils.ui import play, overlay
from splitter.constants import *

import matplotlib.pyplot as plt


left = Box(1600, 1600, 1700, 1800)
middle = Box(1825, 1600, 1975, 1800)
right = Box(2050, 1600, 2175, 1800)

v = VideoStream('/Users/splitter/Downloads/brooklyn.mp4')
v = v[Cut(0,600)][GoodKeyPoints()][ActivityMetric('left', left)][
        ActivityMetric('middle', middle)][ActivityMetric('right', right)][
        Filter('left', [1], 1, delay=25)][
        Filter('middle', [1], 1, delay=25)][
        Filter('right', [1], 1, delay=25)]

count = {'left':0, 'middle':0 ,'right':0}
for p in v:

    count['left'] += p['left']
    count['middle'] += p['middle']
    count['right'] += p['right']

    print(count, p['frame'])

    img = overlay(p['data'], [('', (1600, 1600, 1700, 1800)),  ('', (1850, 1600, 1950, 1800)), ('', (2050, 1600, 2175, 1800))])
    img = overlay(img, p['bounding_boxes'])


    cv2.imshow('Player',cv2.resize(img, (0,0), fx=0.25,fy=0.25))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()


"""
FILE = 'crash1.mp4'
GT = '/Users/splitter/Downloads/LV_v1/VIDS/GT/g' + FILE
VID = '/Users/splitter/Downloads/LV_v1/VIDS/Test/' + FILE

v = VideoStream(GT)
pipeline = v[KeyPoints(blur=1)]

kps = []
for p in pipeline:
    image = p['data']
    kps.append(len(p['bounding_boxes']))
    image = overlay(image, p['bounding_boxes'])

    #print(p['bounding_boxes'])
    #hit q to exit
    #cv2.imshow('Player',image)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    exit()

v = VideoStream(VID)
pipeline = v[MotionVectors()][Speed()]

kps1 = []
tindex = []
for p in pipeline:
    image = p['data']
    #print(image.shape)
    #print(#)

    #mvs = [l for l,b in p['bounding_boxes']]
    #kps1.extend(mvs)

    if len(p['motion_vectors']) > 0:
        v = np.nanmean(p['motion_vectors'])
        kps1.append(v)
    else:
        kps1.append(np.nan)
    #tindex.extend([p['frame']] * len(mvs))

    #print(kps1)

    #kps1.append(len(p['bounding_boxes']))
    #image = overlay(image, p['bounding_boxes'])

from splitter.tracking.ts import *

import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.plot(moving_average(kps1,100))
plt.subplot(2,1,2)
plt.plot(kps)
plt.show()

print(thresh_finder(kps1,100,100))
#print(change_finder(kps,100,20))
"""
