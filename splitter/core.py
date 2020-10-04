"""This file is part of Splitter which is released under MIT License.

core.py defines the basic storage api in splitter.
"""

class StorageManager():
	"""The StorageManager is the basic abstract class that represents a
	storage manager.
	"""
	def __init__(self, content_tagger):
		self.content_tagger = content_tagger

	def put(self, filename, target, args):
		"""putFromFile adds a video to the storage manager from a file
		"""
		raise NotImplemented("putFromFile not implemented")

	def get(self, name, condition, clip_size):
		# NOTE: clip_size is not needed in FullStorageManager
		"""retrieves a clip of a certain size satisfying the condition
		"""
		raise NotImplemented("getIf not implemented")

	def setThreadPool(self):
		raise ValueError("This storage manager does not support threading")

	def delete(self,name):
		raise NotImplemented("delete not implemented")

	def list(self):
		raise NotImplemented("list() not implemented")

	def size(self, name):
		raise NotImplemented("size() not implemented")
