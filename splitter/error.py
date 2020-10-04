
class CorruptedOrMissingVideo(Exception):
   """Raised when opencv cannot open a video"""
   pass

class VideoNotFound(Exception):
   """Video with the specified name not found in the manager"""
   pass

class ManagerIOError(Exception):
   """Unspecified error with the manager"""
   pass

class HeaderError(Exception):
   """Unspecified error with headers"""
   pass

class InvalidRegionError(Exception):
   """The box region specified is invalid"""
   pass