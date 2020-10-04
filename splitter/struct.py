"""This file is part of Splitter which is released under MIT License.

struct.py defines the main data structures used in splitter. It defines a
video input stream as well as operators that can transform this stream.
"""
import cv2
import os
from splitter.error import *
import numpy as np
import json
from timeit import default_timer as timer
import itertools


#sources video from the default camera
DEFAULT_CAMERA = 0


class VideoStream():
	"""The video stream class opens a stream of video
	   from a source.

	Frames are structured in the following way: (1) each frame 
	is a dictionary where frame['data'] is a numpy array representing
	the image content, (2) all the other keys represent derived data.

	All geometric information (detections, countours) go into a list called
	frame['bounding_boxes'] each element of the list is structured as:
	(label, box).
	"""

	def __init__(self, src, limit=-1, origin=np.array((0,0)), offset=0, rows=None, hwang=False):
		"""Constructs a videostream object

		   Input: src- Source camera or file or url
				  limit- Number of frames to pull
				  origin- Set coordinate origin
		"""
		self.src = src
		self.limit = limit
		self.origin = origin
		self.propIds = None
		self.cap = None
		self.time_elapsed = 0
		self.hwang = hwang
		self.finished = False

		# moved from __iter__ to __init__ due to continuous iterating
		if hwang:
			import hwang
			if rows == None:
				cap = cv2.VideoCapture(self.src)
				rows = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
				print(rows)
			self.rows = rows
			self.frame_count = offset
			self.frames = None
			self.decoder = hwang.Decoder(self.src)
		else:
			self.offset = offset
			self.frame_count = offset
			self.cap = cv2.VideoCapture(self.src)


		self.scale = get_scale(src)

	def __getitem__(self, xform):
		"""Applies a transformation to the video stream
		"""
		return xform.apply(self)


	def __iter__(self):
		"""Constructs the iterator object and initializes
		   the iteration state
		"""

		if self.hwang:
			self.width = self.decoder.video_index.frame_width()
			self.height = self.decoder.video_index.frame_height()
			if self.frames == None:
				self.frames = iter(self.decoder.retrieve(self.rows))
		else:
			if self.cap == None:
				# iterate the same videostream again after the previous run has finished
				self.frame_count = self.offset
				self.cap = cv2.VideoCapture(self.src)

			if self.propIds:
				for propId in self.propIds:
					self.cap.set(propId, self.propIds[propId])

			if not self.cap.isOpened():
				raise CorruptedOrMissingVideo(str(self.src) + " is corrupted or missing.")


			#set sizes after the video is opened
			self.width = int(self.cap.get(3))   # float
			self.height = int(self.cap.get(4)) # float
			self.fps = self.cap.get(cv2.CAP_PROP_FPS)

		return self


	def __next__(self):
		if self.finished:
			raise StopIteration("Iterator is closed")

		if self.hwang:
			self.frame_count += 1
			return {'data': next(self.frames),
					'frame': (self.frame_count - 1),
					'origin': self.origin}
		else:
			if self.cap.isOpened() and \
			   (self.limit < 0 or self.frame_count < self.limit):

				time_start = timer()
				ret, frame = self.cap.read()
				self.time_elapsed += timer() - time_start

				if ret:
					self.frame_count += 1
					return {'data': frame, \
							'frame': (self.frame_count - 1),\
							'origin': self.origin}

				else:
					self.cap = None
					raise StopIteration("Iterator is closed")


			else:
				# self.cap.release()  # commented out due to CorruptedOrMissingVideo error
				self.cap = None
				raise StopIteration("Iterator is closed")

	def __call__(self, propIds = None):
		""" Sets the propId argument so that we can
		take advantage of video manipulation already
		supported by VideoCapture (cv2)
		Arguments:
			propIds: {'ID': property}
		"""
		self.propIds = propIds

	def get_cap_info(self, propId):
		""" If we currently have a VideoCapture op
		"""
		if self.cap:
			return self.cap.get(propId)
		else:
			return None

	def lineage(self):
		return [self]	


class IteratorVideoStream(VideoStream):
	"""The video stream class opens a stream of video
	   from an iterator over frames (e.g., a sequence
	   of png files). Compatible with opencv streams.
	"""

	def __init__(self, src, refs, limit=-1):
		"""Constructs a videostream object

		   Input: src- list of vstreams
				  limit- Number of frames to pull
		"""
		self.sources = refs
		self.src = src
		self.limit = limit
		self.global_lineage = [self]

		for ref in refs:
			# if ref is not a filename
			if not isinstance(ref, str):
				self.scale = 1.0
				return

		self.scale = min([get_scale(s) for s in refs], default=1.0)

	def __getitem__(self, xform):
		"""Applies a transformation to the video stream
		"""
		return xform.apply(self)

	def __iter__(self):
		"""Constructs the iterator object and initializes
		   the iteration state
		"""

		try:
			self.frame_iter = iter(self.src)
		except:
			raise CorruptedOrMissingVideo(str(self.src) + " is corrupted or missing.")

		try:
			self.next_frame = next(self.frame_iter)
			# set sizes after the video is opened
			if 'data' in self.next_frame:
				self.width = int(self.next_frame['data'].shape[0])  # float
				self.height = int(self.next_frame['data'].shape[1])  # float

			self.frame_count = 1
		except StopIteration:
			self.next_frame = None

		return self

	def __next__(self):
		if self.next_frame == None:
			raise StopIteration("Iterator is closed")

		if (self.limit < 0 or self.frame_count <= self.limit):
			ret = self.next_frame
			self.next_frame = next(self.frame_iter)

			if 'frame' in ret:
				return ret
			else:
				self.frame_count += 1
				ret.update({'frame': (self.frame_count - 1)})
				return ret
		else:
			raise StopIteration("Iterator is closed")

	def lineage(self):
		return self.global_lineage


class RawVideoStream(VideoStream):
	"""The video stream class opens a stream of video
	   from an iterator over decoded, serialized frames
	"""

	def __init__(self, src, shape, limit=-1, \
				 origin=np.array((0,0)), offset=0, buffer_size=10):
		"""Constructs a videostream object

		   Input: src- iterator over frames
				  limit- Number of frames to pull
		"""
		self.src = src
		self.shape = shape
		self.limit = limit

		self.global_lineage = []
		self.origin = origin
		self.offset = offset

		self.mmap_colors = None
		self.buffer_size = buffer_size

	def __getitem__(self, xform):
		"""Applies a transformation to the video stream
		"""
		return xform.apply(self)

	def __iter__(self):
		"""Constructs the iterator object and initializes
		   the iteration state
		"""
		#np.memmap(self.src, dtype='uint8', mode='r', shape=self.shape)

		try:
			self.frame_iter = np.memmap(self.src, 
										dtype='uint8', \
										mode='r', \
										shape=self.shape,
										order='F')

			self.buffer = None
		except:
			raise CorruptedOrMissingVideo(str(self.src) + " is corrupted or missing.")

		try:
			# set sizes after the video is opened
			self.width = self.shape[1] #int(self.next_frame.shape[0])  # float
			self.height = self.shape[2] #int(self.next_frame.shape[1])  # float
			self.frame_count = self.offset
		except StopIteration:
			self.next_frame = None

		return self

	def __next__(self):

		index = self.frame_count-self.offset
		buffer_index = (index % self.buffer_size)

		if index >= self.shape[0]:
			raise StopIteration("Iterator is closed")
		elif buffer_index >= self.frame_iter.shape[0]:
			raise StopIteration("Iterator is closed")
		elif buffer_index == 0:
			self.buffer = self.frame_iter[index:index+self.buffer_size,:,:,:]

		self.frame_count += 1
		#ret = .copy()

		return {'frame': (self.frame_count - 1), 'data': self.buffer[buffer_index,:,:,:], 'origin': self.origin}

	def lineage(self):
		return self.global_lineage



#helper methods
def getFrameData(frame):
	return frame['data']

def getFrameNumber(frame):
	return frame['frame']

def getFrameMetaData(frame):
	return {k:frame[k] for k in frame if k != 'data'}

#given a list of pipeline methods, it reconstucts it into a stream
def build(lineage):
	"""build(lineage) takes as input the lineage of a stream and
	constructs the stream.
	"""
	plan = lineage
	if len(plan) == 0:
		raise ValueError("Plan is empty")
	elif len(plan) == 1:
		return plan[0]
	else:
		v = plan[0]
		for op in plan[1:]:
			v = v[op]
		return v


class Operator():
	"""An operator defines consumes an iterator over frames
	and produces and iterator over frames. The Operator class
	is the abstract class of all pipeline components in dlcv.
	
	We overload python subscripting to construct a pipeline
	>> stream[Transform()] 
	"""

	#this is a function that sets some bookkeepping variables
	#that are useful for sizing streams without opening the frames.
	def super_iter(self):
		self.width = self.video_stream.width
		self.height = self.video_stream.height
		self.frame_count = self.video_stream.frame_count
		self.fps = self.video_stream.fps

	def super_next(self):
		self.frame_count = self.video_stream.frame_count

	#subscripting binds a transformation to the current stream
	def apply(self, vstream):
		self.video_stream = vstream
		return self

	def __getitem__(self, xform):
		"""Applies a transformation to the video stream
		"""
		return xform.apply(self)

	def lineage(self):
		"""lineage() returns the sequence of transformations
		that produces the given stream of data. It can be run
		without materializing any of the stream.

		Output: List of references to the pipeline components
		"""
		if isinstance(self.video_stream, VideoStream):
			return [self.video_stream, self]
		else:
			return self.video_stream.lineage() + [self]

	def _serialize(self):
		return NotImplemented("This operator cannot be serialized")

	def serialize(self):
		try:
			import json
			return json.dumps(self._serialize())
		except:
			return ManagerIOError("Serialization Error")




class Box():
	"""Bounding boxes are a core geometric construct in the dlcv
	system. The Box() class defines a bounding box with named coordinates
	and manipulation methods to determine intersection and containment.
	"""

	def __init__(self,x0,y0,x1,y1):
		"""The constructor for a box, all of the inputs have to be castable to 
		integers. By convention x0 <= x1 and y0 <= y1
		"""

		self.x0 = int(x0)
		self.y0 = int(y0)
		self.x1 = int(x1)
		self.y1 = int(y1)

		if x0 > x1 or y0 > y1:
			raise InvalidRegionError("The specified box is invalid: " + str([x0,y0,x1,y1]))

	#shifts the box to a new origin
	def shift(self, origin):
		return Box(self.x0-origin[0], \
				   self.y0-origin[1], \
				   self.x1-origin[0], \
				   self.y1-origin[1])

	def x_translate(self, x):
		return Box(self.x0 + x, \
				   self.y0, \
				   self.x1 + x, \
				   self.y1)
				   
	def y_translate(self, y):
		return Box(self.x0, \
				   self.y0 + y, \
				   self.x1, \
				   self.y1 + y)
	def area(self):
		"""Calculates the area contained in the box
		"""
		return np.abs((self.x1 - self.x0) * (self.y1 - self.y0))

	def __mul__(self, scalar):
		return Box(int(self.x0/scalar), \
				   int(self.y0/scalar), \
				   int(self.x1*scalar), \
				   int(self.y1*scalar))

	def __add__(self, scalar):
		return Box(int(self.x0 - scalar), \
				   int(self.y0 - scalar), \
				   int(self.x1 + scalar), \
				   int(self.y1 + scalar))

	#helpher methods to test intersection and containement
	def _zero_x_cond(self, other):
		return (other.x0 >= self.x0 and other.x0 <= self.x1)

	def _zero_y_cond(self, other):
		return (other.y0 >= self.y0 and other.y0 <= self.y1)

	def _one_x_cond(self, other):
		return (other.x1 >= self.x0 and other.x1 <= self.x1)

	def _one_y_cond(self, other):
		return (other.y1 >= self.y0 and other.y1 <= self.y1)

	"""Intersection and containement
	"""
	def contains(self, other):
		return self._zero_x_cond(other) and \
			   self._zero_y_cond(other) and \
			   self._one_x_cond(other) and \
			   self._one_y_cond(other)

	def intersect(self, other):
		x = self.x1 >= other.x0 and other.x1 >= self.x0
		y = self.y1 >= other.y0 and other.y1 >= self.y0
		return x and y

	def intersect_area(self, other):
		if self.intersect(other):
			x = min(self.x1, other.x1) - max(self.x0, other.x0)
			if x < 0:
				x = 0
			y = min(self.y1, other.y1) - max(self.y0, other.y0)
			if y < 0:
				y = 0
			return x*y
		else:
			return 0

	def union_area(self, other):
		ia = self.intersect_area(other)
		return self.area() + other.area() - ia
	
	def union_box(self, other):
		#print('union',self, other)
		return Box(min(self.x0, other.x0), \
				min(self.y0, other.y0), \
				max(self.x1, other.x1), \
				max(self.y1, other.y1))
	"""The storage manager needs a tuple representation of the box, this serializes it.
	"""
	def serialize(self):
		return int(self.x0),int(self.y0),int(self.x1),int(self.y1)


	def __str__(self):
		return str(self.serialize())

	__repr__ = __str__


class CustomTagger(Operator):
	def __init__(self, tagger, batch_size=-1):
		super(CustomTagger, self).__init__()
		# a custom tagger function that takes video_stream and batch_size; it raises StopIteration when finishes
		self.tagger = tagger
		self.batch_size = batch_size
		self.next_count = 0  # how many next() we have called after _get_tags()
		self.stream = False
		self.frames = []

	def __iter__(self):
		self.input_iter = iter(self.video_stream)
		self.super_iter()
		if self.batch_size == -1:
			self.batch_size = self.fps
		return self

	def _get_tags(self):
		if self.next_count == 0 or self.next_count >= self.batch_size:
			self.next_count = 0
			# we assume it iterates the entire batch size and save the results
			self.tags = []
			try:
				if self.stream:
					tag, frames = self.tagger(self.input_iter, self.batch_size, video = True)
					self.frames = frames
				else:
					tag = self.tagger(self.input_iter, self.batch_size)
			except StopIteration:
				raise StopIteration("Iterator is closed")
			if tag:
				self.tags.append(tag)

		self.next_count += 1
		return self.tags

	def __next__(self):
		if self.stream:
			tags = self._get_tags()
			return {'objects': tags, 'frame': self.frames[self.next_count - 1]}
		else:
			return {'objects': self._get_tags()}

	def set_stream(self, stream):
		self.stream = stream

class Serializer(json.JSONEncoder):
	def default(self, obj):
		return obj.serialize()

def get_scale(file):
	filename, file_extension = os.path.splitext(file)
	prefix = filename.split('-')

	if len(prefix) == 1:
		return 1.0
	else:
		try:
			return float(prefix[-1])
		except:
			return 1.0
