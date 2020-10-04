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

#from splitter.extern.ffmpeg import *
#from splitter

manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=30), CropSplitter(), 'videos')
#manager = FullStorageManager(SizeMovementTagger(), CropSplitter(), 'videos')
#manager = FullStorageManager(TensorFlowObjectDetect(model_file='ssd_mobilenet_v1_coco_2017_11_17', label_file='mscoco_label_map.pbtxt',
#                               num_classes=90, confidence=0.25), CropSplitter(), 'videos')
manager.put('./cut3.mp4', 'test2')
#res = manager.get('test', Condition(label='small'))
#print([video for video in res[0]][0]['data'].shape)
#print(len(res))
#play(res[0])


#manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=300), CropSplitter(), 'videos')
#manager.put('tcam.mp4', 'test', args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': 1000, 'batch_size': 100, 'num_processes': 4, 'background_scale': 1})
#clips = manager.get('test', Condition(label='foreground'))


"""
for i in range(25,60,3):
	new_file = set_bitrate('tcam.mp4',str(i)+'tcam.avi', i)
	c = VideoStream(new_file, limit=1000)
	region = Box(200,550,350,750)
	pipelines = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
	result = count(pipelines, ['one'], stats=True)
	print('Bitrate',i,result)
"""

from splitter.optimizer.splitter import SplitterOptimizer

for i in range(10,1,-1):
	scale = i/10.0
	new_file = set_quality('tcam.mp4','tcam-'+str(scale)+".avi",25, scale)
	c = VideoStream('tcam-'+str(scale)+".avi", limit=1000)
	region = Box(200,550,350,750)



	#print(scale, )
	#region2 = Box(region.x0*scale, region.y0*scale, region.x1*scale, region.y1*scale)
	#region3 = Box(region.x0*scale*1.1, region.y0*scale*1.1, region.x1*scale*1.1, region.y1*scale*1.1)
	#scale = get_scale('tcam-'+str(scale)+".avi")

	d = SplitterOptimizer(adaptive_blur=True)
	pipelines = c[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
	d.optimize(pipelines)

	result = count(pipelines, ['one'], stats=True)
	print('Resolution',scale, result)


"""
for i in range(25,60,3):
	region = Box(200,550,350,750)
	from splitter.optimizer.splitter import SplitterOptimizer
	d = SplitterOptimizer(adaptive_blur=True)
	v = VideoStream(str(i)+'tcam.avi', limit=1000)
	v = v[KeyPoints()][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
	#c = IteratorVideoStream(itertools.chain(*v), v)
	d.optimize(v)
	print('Scale: ' + str(i),count(v, ['one'], stats=True))
"""

#pipelines = c[KeyPoints(blur=1)][ActivityMetric('one', region)][Filter('one', [-0.25,-0.25,1,-0.25,-0.25],1.5, delay=10)]
#print(count(pipelines, ['one'], stats=True))