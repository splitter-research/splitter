"""This file is part of Splitter which is released under MIT License.

The SimpleStorageManager class acts as a default baseline for the splitter
system. It provides a basic file io and network io interface to put and get
videos into the storage system. 
"""
from splitter.core import *
from splitter.constants import *
from splitter.struct import *
from splitter.simple_manager.videoio import *
from splitter.simple_manager.file import *
from splitter.deprecated_header import *
from splitter.dataflow.map import *
from splitter.error import *

import os
import logging


class SimpleStorageManager(StorageManager):
	"""The SimpleStorageManger stores videos as files that are temporally partitioned
	   into equiwidth segments.
	"""

	DEFAULT_ARGS = {'encoding': GSC, 'limit': -1, 'sample': 1.0, 'offset': 0, 'batch_size': 20}


	def __init__(self, basedir):
		'''Every simplestoragemanager takes as input  a 
		   basedir for storage.
		'''
		self.basedir = basedir
		self.videos = set()

		if not os.path.exists(basedir):
			try:
				os.makedirs(basedir)
			except:
				raise ManagerIOError("Cannot create the directory: " + str(basedir))


	def put(self, filename, target, args=DEFAULT_ARGS):
		'''Put takes in a file on disk and writes it as the target name with the arguments
		'''
		self.doPut(filename, target, args)


	def get(self, name, condition, args=DEFAULT_ARGS):
		'''Get takes in a name and a condition that the system tries a best effort to push down
		'''
		logging.info("Calling get()")
		return self.doGet(name, condition, args['batch_size'])


	def delete(self, name):
		'''Delete deletes a clip from the storage engine
		'''
		physical_clip = os.path.join(self.basedir, name)

		if name in self.videos:
			self.videos.remove(name)

		delete_video_if_exists(physical_clip)



	def list(self):
		'''List lists all the clips in the engine
		'''
		return list(self.videos)


	def size(self, name):
		'''Returns the storage size of a clip
		'''
		seq = 0
		size = 0
		physical_clip = os.path.join(self.basedir, name)

		while True:

			try:
				file = add_ext(physical_clip, '.seq', seq) 
				size += sum(os.path.getsize(os.path.join(file,f)) for f in os.listdir(file))
				seq += 1

			except FileNotFoundError:
				break

		return size


	

	def doPut(self, filename, target, args=DEFAULT_ARGS):
		"""putFromFile adds a video to the storage manager from a file
		"""
		v = VideoStream(filename, args['limit'])
		v = v[Sample(args['sample'])]

		physical_clip = os.path.join(self.basedir, target)
		delete_video_if_exists(physical_clip)

		if args['batch_size'] == -1:
			write_video(v, \
					    physical_clip, args['encoding'], \
					    ObjectHeader(offset=args['offset']))
		else:
			write_video_clips(v, \
							  physical_clip, \
							  args['encoding'], \
							  ObjectHeader(offset=args['offset']), \
							  args['batch_size'])

		self.videos.add(target)

	


	def doGet(self, name, condition, clip_size):
		"""retrieves a clip of a certain size satisfying the condition
		"""
		physical_clip = os.path.join(self.basedir, name)

		try:
			return read_if(physical_clip, condition, clip_size)
		# If a file is not found, it indicates that the clip is not in the storage
		# manager. We should move along and raise a VideoNotFound error so that the user
		# gets a more descriptive error
		except FileNotFoundError:
			pass
		except:
			raise

		raise VideoNotFound(name + " not found in storage manager")
