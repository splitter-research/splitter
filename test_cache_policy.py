from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.tracking.background import FixedCameraBGFGSegmenter
from splitter.tracking.contour import *
from splitter.dataflow.map import *
from splitter.full_manager.full_manager import *
from splitter.full_manager.video_processing_keypoint import *
from splitter.object_detection.detect import *

from splitter.dataflow.agg import *
from splitter.tracking.contour import *
from splitter.tracking.event import *

from splitter.optimizer.splitter import SplitterOptimizer

def time_filter(start, end):

	def do_filter(conn, video_name):
		c = conn.cursor()
		c.execute("SELECT clip_id FROM clip WHERE ((start_time > %s AND start_time < %s) OR (end_time > %s AND end_time < %s) ) AND video_name = '%s'" % (str(start),str(end), str(start),str(end), video_name))
		return [cl[0] for cl in c.fetchall()]

	return do_filter

def spatial_selectivity(buffer=0):
	return Box(275-buffer, 575-buffer, 325+buffer, 625+buffer)

def cleanUp():
	if os.path.exists('./videos'):
		shutil.rmtree('./videos')

def doexperiments_lru(budget=1000000000):

	for i in range(0,250,50):
		cleanUp()

		manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=100), CropSplitter(), 'videos')
		manager.put('tcam.mp4', 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': 2000, 'batch_size': 100, 'num_processes': 4, 'background_scale': 1})
		clips = manager.get('test', Condition(label='foreground'))

		region = spatial_selectivity(buffer=i)

		d = SplitterOptimizer()
		pipelines = []
		for c in clips:
			pipeline = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
			pipeline = d.optimize(pipeline)
			pipelines.append(pipeline)

		result, time1 = counts(pipelines, ['one'], stats=True) 
		d.cacheLRU(manager, budget)

		clips = manager.get('test', Condition(label='foreground'))
		pipelines = []
		for c in clips:
			pipeline = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
			pipeline = d.optimize(pipeline)
			pipelines.append(pipeline)

		result, time2 = counts(pipelines, ['one'], stats=True) 

		print(i, budget, time1['elapsed'], time2['elapsed'])

def doexperiments_ks(budget=1000000000):

	for i in range(0,300,50):
		cleanUp()

		manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=100), CropSplitter(), 'videos')
		manager.put('tcam.mp4', 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': 2000, 'batch_size': 100, 'num_processes': 4, 'background_scale': 1})
		clips = manager.get('test', Condition(label='foreground'))

		region = spatial_selectivity(buffer=i)

		d = SplitterOptimizer()
		pipelines = []
		for c in clips:
			pipeline = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
			pipeline = d.optimize(pipeline)
			pipelines.append(pipeline)

		result, time1 = counts(pipelines, ['one'], stats=True) 
		d.cacheKnapsack(manager, budget)

		clips = manager.get('test', Condition(label='foreground'))
		pipelines = []
		for c in clips:
			pipeline = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
			pipeline = d.optimize(pipeline)
			pipelines.append(pipeline)

		result, time2 = counts(pipelines, ['one'], stats=True) 

		print(i, budget, time1['elapsed'], time2['elapsed'])

doexperiments_ks(budget=1000000000)
doexperiments_lru(budget=1000000000)
#doexperiments_lru(budget=5000000000)
