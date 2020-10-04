"""This file is part of Splitter which is released under MIT License.

background.py defines some background-foreground separation routines that
are useful for processing fixed camera videos.
"""

import cv2
import numpy as np
from splitter.struct import Box

#fixed camera bg segmentation
#finds "moving" pixels
class FixedCameraBGFGSegmenter(object):

	def __init__(self, movement_threshold=25, #bigger means that movements needs to be more significant
				 blur=21, #reduces noise
				 movement_prob=0.05): #each pixel has a probability of movement.

		self.movement_threshold = movement_threshold
		self.blur = blur
		self.movement_prob = movement_prob


	#returns a bounding box around the foreground.
	def segment(self, vstream, batch_size, video = False):

		dynamic_mask = None
		prev = None
		count = 0
		frames = []
		for frame in vstream:
			img = frame['data']
			#print(frame['frame'])
			if video:
				frames.append(img)

			if len(img.shape) < 3:
				gray = img
			else:
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

			blurred = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)

			if not (prev is None):

				frameDelta = cv2.absdiff(blurred, prev)
				thresh = cv2.threshold(frameDelta, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]

				if dynamic_mask is None:
					dynamic_mask = thresh.astype(np.float32)
				else:
					dynamic_mask += thresh.astype(np.float32)/255


			prev = blurred
			count += 1

			if count >= batch_size:
				break

		if count == 0:
			raise StopIteration("Iterator is closed")

		cthresh = count * self.movement_prob
		#print(cthresh, np.max(dynamic_mask))

		#print(np.max(dynamic_mask))

		try:
			y0 =  np.min(np.argwhere(dynamic_mask >= cthresh)[:,0])
			x0 =  np.min(np.argwhere(dynamic_mask >= cthresh)[:,1])
			y1 =  np.max(np.argwhere(dynamic_mask >= cthresh)[:,0])
			x1 =  np.max(np.argwhere(dynamic_mask >= cthresh)[:,1])

			#print(x0, y0, x1, y1, cthresh)

			if np.abs(x0-x1) <= 0 or np.abs(y0-y1) <= 0:
				return {}

			#flipped axis in crop
			#print('box',x0, y0, x1, y1)

			#print(x0, y0, x1, y1)

			if video:
				return {'label': 'foreground', 'bb': Box(x0, y0, x1, y1)}, frames

			return {'label': 'foreground', 'bb': Box(x0, y0, x1, y1)}

		except:
			if video:
				return {}, frames
			return {}