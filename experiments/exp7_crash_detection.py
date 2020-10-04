import sys
import environ

from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.tracking.background import FixedCameraBGFGSegmenter
from splitter.optimizer.splitter import SplitterOptimizer
from splitter.utils.testing_utils import get_size
from experiments.environ import logrecord

from splitter.full_manager.full_manager import *
from splitter.dataflow.agg import *
from splitter.tracking.contour import *
from splitter.tracking.event import *
from splitter.simple_manager.manager import *

import datetime
from splitter.dataflow.xform import *

FOLDER = '/Users/splitter/Downloads/LV_v1/VIDS/Test/'

def pipeline_get(p):
	for i in p:
		#print(p.height,p.width)
		pass


def naive(index=0):

	if os.path.exists('videos'):
		shutil.rmtree('videos')

	now = datetime.datetime.now()

	filename = FOLDER + 'crash{0}.mp4'.format(index)

	v = VideoStream(filename)
	pipeline = v[MotionVectors()][Speed()]

	pipeline_get(pipeline)

	result = (datetime.datetime.now() - now).total_seconds()

	logrecord('naive', ({'folder_size': os.path.getsize(filename) }), 'get', str(result), 's')


def full(index=0):

	if os.path.exists('videos'):
		shutil.rmtree('videos')

	#11,11
	manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter(blur=11,movement_threshold=5).segment, batch_size=200), CropSplitter(), 'videos')

	now = datetime.datetime.now()

	filename = FOLDER + 'crash{0}.mp4'.format(index)

	manager.put(filename, 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': -1, 'batch_size': 200,
                      'num_processes': 4, 'background_scale': 1}, hwang=False)

	put_result = (datetime.datetime.now() - now).total_seconds()

	logrecord('full', ({'folder_size': get_size('videos') }), 'put', str(put_result), 's')

	now = datetime.datetime.now()

	clips = manager.get('test', Condition(label='foreground'))
	for c in clips:
		pipeline_get(c[MotionVectors()][Speed()])

	result = (datetime.datetime.now() - now).total_seconds()

	logrecord('full', ({'folder_size': get_size('videos') }), 'get', str(result), 's')


def fullQuality(index=0):

	if os.path.exists('videos'):
		shutil.rmtree('videos')

	manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter(blur=11,movement_threshold=11).segment, batch_size=200), CropSplitter(), 'videos')

	now = datetime.datetime.now()

	filename = FOLDER + 'crash{0}.mp4'.format(index)

	manager.put(filename, 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit':-1, 'batch_size': 200,
                      'num_processes': 4, 'background_scale': 1}, hwang=False)

	#no queries to the background, lowest quality for speed to be within 5%
	manager.set_quality('test', Condition(label='background'), qscale=52, rscale=0.1)
	manager.set_quality('test', Condition(label='foreground'), qscale=44, rscale=1)

	put_result = (datetime.datetime.now() - now).total_seconds()

	logrecord('fullq', ({'folder_size': get_size('videos') }), 'put', str(put_result), 's')

	now = datetime.datetime.now()

	clips = manager.get('test', Condition(label='foreground'))
	for c in clips:
		pipeline_get(c[MotionVectors()][Speed()])

	result = (datetime.datetime.now() - now).total_seconds()

	logrecord('fullq', ({'folder_size': get_size('videos') }), 'get', str(result), 's')


for i in range(0,6):
	naive(i)
	full(i)
	fullQuality(i)