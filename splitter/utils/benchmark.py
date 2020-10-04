from splitter.constants import *
from splitter.core import *
from splitter.utils.testing_utils import *
import json

class PerformanceTest():

	def __init__(self, storage_manager, test_video):
		self.sm = storage_manager
		self.test_video = test_video

	def putEncodingOneMin(self):
		for enc in ENCODINGS:
			args = {'encoding': enc, 'size': -1, 'limit': 60*DEFAULT_FRAME_RATE, 'sample': 1.0, 'offset': 0}

			#time put
			now = time.time()
			self.sm.put(self.test_video, 'test', args)
			time_result = (time.time() - now)
			size_result = self.sm.size('test')

			log = {'time': time_result, 'space': size_result}
			log.update(args)

			self.sm.delete('test')

			print(json.dumps(log))


	def putClipOneMin(self):
		for size in range(5,60,5):
			args = {'encoding': MP4V, 'size': size*DEFAULT_FRAME_RATE, 'limit': 60*DEFAULT_FRAME_RATE, 'sample': 1.0, 'offset': 0}

			#time put
			now = time.time()
			self.sm.put(self.test_video, 'test', args)
			time_result = (time.time() - now)
			size_result = self.sm.size('test')

			log = {'time': time_result, 'space': size_result}
			log.update(args)

			self.sm.delete('test')

			print(json.dumps(log))


	def getClipSizeOneMin(self):
		for size in range(5,60,5):
			args = {'encoding': MP4V, 'size': -1, 'limit': 60*DEFAULT_FRAME_RATE, 'sample': 1.0, 'offset': 0}

			#time put
			self.sm.put(self.test_video, 'test', args)

			now = time.time()
			time_result = timeof(self.sm.get('test', Condition(filter=TRUE), int(size*DEFAULT_FRAME_RATE)))
			full_time_result = (time.time() - now)
			log = {'time': time_result,'first_frame': full_time_result - time_result, 'retr_clip_size': size}
			log.update(args)

			self.sm.delete('test')

			print(json.dumps(log))


	def getClipSizeTenSec(self):
		for size in range(5,60,5):
			args = {'encoding': MP4V, 'size': size, 'limit': 60*DEFAULT_FRAME_RATE, 'sample': 1.0, 'offset': 0}

			#time put
			self.sm.put(self.test_video, 'test', args)

			now = time.time()
			time_result = timeof(self.sm.get('test', Condition(filter=TRUE), int(10*DEFAULT_FRAME_RATE)))
			full_time_result = (time.time() - now)
			log = {'time': time_result,'first_frame': full_time_result - time_result, 'retr_clip_size': 10}
			log.update(args)

			self.sm.delete('test')

			print(json.dumps(log))


	def getEncTenTenSec(self):
		for enc in ENCODINGS:
			args = {'encoding': enc, 'size': 10*DEFAULT_FRAME_RATE, 'limit': 60*DEFAULT_FRAME_RATE, 'sample': 1.0, 'offset': 0}

			#time put
			self.sm.put(self.test_video, 'test', args)

			now = time.time()
			time_result = timeof(self.sm.get('test', Condition(filter=TRUE), int(10*DEFAULT_FRAME_RATE)))
			full_time_result = (time.time() - now)
			log = {'time': time_result,'first_frame': full_time_result - time_result, 'retr_clip_size': 10}
			log.update(args)

			self.sm.delete('test')

			print(json.dumps(log))


	def getSelTenTenSec(self):
		for size in range(0,70,10):
			args = {'encoding': MP4V, 'size': 10*DEFAULT_FRAME_RATE, 'limit': 60*DEFAULT_FRAME_RATE, 'sample': 1.0, 'offset': 0}

			#time put
			self.sm.put(self.test_video, 'test', args)
			now = time.time()
			time_result = timeof(self.sm.get('test', Condition(filter=startsBefore(size*DEFAULT_FRAME_RATE)), int(10*DEFAULT_FRAME_RATE)))
			full_time_result = (time.time() - now)
			log = {'time': time_result,'first_frame': full_time_result - time_result, 'retr_clip_size': size, 'sel': size/60}
			log.update(args)

			self.sm.delete('test')

			print(json.dumps(log))

	def runAll(self):
		print('[dlstorage] put() for different encodings and a video of 1 min')
		self.putEncodingOneMin()
		print('[dlstorage] put() for MP4V and varying clip size for a video of 1 min')
		self.putClipOneMin()
		print('[dlstorage] get() for MP4V full video of different clip sizes')
		self.getClipSizeOneMin()
		print('[dlstorage] get() for MP4V 10 sec clips of different get clip sizes')
		self.getClipSizeTenSec()
		print('[dlstorage] get() for different encodings 10 sec clips of different 10 sec sizes')
		self.getEncTenTenSec()
		print('[dlstorage] get() for different selectivities 10 sec clips of different 10 sec sizes')
		self.getSelTenTenSec()


