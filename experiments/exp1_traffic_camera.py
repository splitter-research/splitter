import environ
import sys
from environ import *

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


#loads directly from the mp4 file
def runNaive(src, tot=1000, sel=0.1):
	cleanUp()

	c = VideoStream(src, limit=tot)
	sel = sel/2
	region = Box(200,550,350,750)
	pipelines = c[Cut(tot//2-int(tot*sel),tot//2+int(tot*sel))][KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
	result = count(pipelines, ['one'], stats=True)

	logrecord('naive',({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')


#Simple storage manager with temporal filters
def runSimple(src, tot=1000, sel=0.1):
	cleanUp()

	manager = SimpleStorageManager('videos')
	now = timer()
	manager.put(src, 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 20, 'num_processes': 4, 'background_scale': 1})
	put_time = timer() - now
	print("Put time for simple:", put_time)

	region = Box(200,550,350,750)

	sel = sel/2

	clips = manager.get('test',lambda f: overlap(f['start'],f['end'],tot//2-int(tot*sel),tot//2+int(tot*sel)))
	pipelines = []
	for c in clips:
		pipelines.append(c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)])

	result = counts(pipelines, ['one'], stats=True)

	logrecord('simple',({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')


#Full storage manager with bg-fg optimization
def runFull(src, tot=1000, sel=0.1):
	cleanUp()

	manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=100), CropSplitter(), 'videos')
	now = timer()
	manager.put(src, 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 100, 'num_processes': 4, 'background_scale': 1})
	put_time = timer() - now
	print("Put time for simple:", put_time)

	region = Box(200, 550, 350, 750)
	sel = sel/2

	clips = manager.get('test', Condition(label='foreground', custom_filter=time_filter(tot//2-int(tot*sel),tot//2+int(tot*sel))))
	pipelines = []

	for c in clips:
		pipelines.append(c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)])

	result = counts(pipelines, ['one'], stats=True)

	logrecord('full',({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')



#All optimizations
def runFullOpt(src, tot=1000, sel=0.1):
	cleanUp()

	manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=20), CropSplitter(), 'videos')
	now = timer()
	manager.put(src, 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 20, 'num_processes': 4, 'background_scale': 1})
	put_time = timer() - now
	print("Put time for simple:", put_time)

	region = Box(200, 550, 350, 750)
	sel = sel/2
	
	clips = manager.get('test', Condition(label='foreground',custom_filter=time_filter(tot//2-int(tot*sel),tot//2+int(tot*sel))))

	pipelines = []
	d = SplitterOptimizer()
	for c in clips:
		pipeline = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
		pipeline = d.optimize(pipeline)
		pipelines.append(pipeline)

	result = counts(pipelines, ['one'], stats=True)

	logrecord('fullopt',({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
do_experiments(sys.argv[1], [runNaive, runSimple, runFull, runFullOpt], 1000, range(2,10))