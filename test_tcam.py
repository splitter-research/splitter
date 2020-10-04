import sys
from experiments.environ import *

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

import cv2
import numpy as np


# loads directly from the mp4 file
def runNaive(src, tot=-1, sel=0.1):
    cleanUp()

    c = VideoStream(src, limit=tot)
    sel = sel / 2
    region = Box(515, 200, 700, 600)
    pipelines = \
    c[KeyPoints()][ActivityMetric('one', region)][
        Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)]
    result = count(pipelines, ['one'], stats=True)

    logrecord('naive', ({'file': src}), 'get', str(result), 's')

def runNaiveOpt(src, tot=-1, sel=0.1):
    cleanUp()

    c = VideoStream(src, limit=tot)
    sel = sel / 2
    region = Box(515, 200, 700, 600)
    pipelines = \
    c[KeyPoints()][ActivityMetric('one', region)][
        Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)]
    d = SplitterOptimizer()
    pipelines = d.optimize(pipelines)
    result = count(pipelines, ['one'], stats=True)

    logrecord('naive', ({'file': src}), 'get', str(result), 's')


# Simple storage manager with temporal filters
def runSimple(src, tot=-1, sel=0.1):
    cleanUp()

    manager = SimpleStorageManager('videos')
    now = timer()
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 100,
                      'num_processes': 4})
    put_time = timer() - now
    print("Put time for simple:", put_time)

    region = Box(515, 200, 700, 600)

    sel = sel / 2

    clips = manager.get('test', lambda f: True)
    pipelines = []
    for c in clips:
        pipelines.append(c[KeyPoints()][ActivityMetric('one', region)][
                             Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)])

    result = counts(pipelines, ['one'], stats=True)

    logrecord('simple', ({'file': src}), 'get', str(result), 's')


# Simple storage manager with temporal filters
def runSimpleOpt(src, tot=-1, sel=0.1):
    cleanUp()

    manager = SimpleStorageManager('videos')
    now = timer()
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 100,
                      'num_processes': 4})
    put_time = timer() - now
    print("Put time for simple:", put_time)

    region = Box(515, 200, 700, 600)

    sel = sel / 2

    clips = manager.get('test', lambda f: True)
    pipelines = []
    for c in clips:
        pipeline = c[KeyPoints()][ActivityMetric('one', region)][
                             Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)]
        d = SplitterOptimizer()
        pipelines.append(d.optimize(pipeline))

    result = counts(pipelines, ['one'], stats=True)

    logrecord('simple', ({'file': src}), 'get', str(result), 's')

# Full storage manager with bg-fg optimization
def runFull(src, tot=-1, sel=0.1):
    cleanUp()

    manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=100), CropSplitter(),
                                 'videos')
    now = timer()
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 100,
                      'num_processes': 12})
    put_time = timer() - now
    print("Put time for simple:", put_time)

    region = Box(515, 200, 700, 600)
    sel = sel / 2

    clips = manager.get('test', Condition(label='foreground'))
    pipelines = []

    for c in clips:
        pipelines.append(c[KeyPoints()][ActivityMetric('one', region)][
                             Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)])

    result = counts(pipelines, ['one'], stats=True)

    logrecord('full', ({'file': src}), 'get', str(result), 's')


# All optimizations
def runFullOpt(src, tot=-1, sel=0.1):
    cleanUp()

    manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=100), CropSplitter(),
                                 'vi deos')
    now = timer()
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 100,
                      'num_processes': 12})
    put_time = timer() - now
    print("Put time for simple:", put_time)

    region = Box(515, 200, 700, 600)
    sel = sel / 2

    clips = manager.get('test', Condition(label='foreground'))

    pipelines = []
    d = SplitterOptimizer()
    for c in clips:
        pipeline = c[KeyPoints()][ActivityMetric('one', region)][
            Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)]
        pipeline = d.optimize(pipeline)
        pipelines.append(pipeline)

    result = counts(pipelines, ['one'], stats=True)

    logrecord('fullopt', ({'file': src}), 'get', str(result), 's')


#logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
#do_experiments(sys.argv[1], [runSimpleOpt], -1, [1])

from splitter.dataflow.xform import *
from splitter.utils.ui import play, overlay
from splitter.extern.vehicle import VehicleType

v = VideoStream('/Users/splitter/Dropbox/tcam.mp4')

region = Box(475, 500, 700, 700)
pipeline = v[KeyPoints()]\
            [ActivityMetric('one', region)]\
            [Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=5)]\
            [VehicleType('one','type', region)]


prev = "None"
for p in pipeline:

    img = p['data']

    if p['type']:
        prev = p['type']

    cv2.rectangle(img, (region.x0,region.y0), (region.x1,region.y1),(0,255,0), 4)
    cv2.putText(img, str(prev), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 4) 

    cv2.imshow('Player',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        continue



