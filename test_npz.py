import sys


from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.tracking.background import FixedCameraBGFGSegmenter
from splitter.optimizer.splitter import SplitterOptimizer

from splitter.struct import *
from splitter.utils import *
from splitter.dataflow.map import *
from splitter.full_manager.full_manager import *
from splitter.utils.testing_utils import *
from splitter.dataflow.agg import *
from splitter.tracking.contour import *
from splitter.tracking.event import *
from splitter.core import *
from splitter.simple_manager.manager import *

from splitter.utils.ui import play

import cv2
import numpy as np

from splitter.extern.cache import persist

FILENAME = '' #the video file that you want to load
vstream = VideoStream(FILENAME, limit=1000) #limit is the max number of frames

size = persist(vstream, 'cache.npz') #how big the size of the stored raw video is

from splitter.struct import RawVideoStream

vstream = RawVideoStream('cache.npz', shape=(1000,1080,1920,3)) #retrieving the data (have to provide dimensions (num frames, w, h, channels)

#do something
for v in vstream:
    pass

