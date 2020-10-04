"""This file is part of Splitter which is released under MIT License.

This file contains a bunch of primitives for manipulating clip boundaries.
"""

import itertools
import copy

from splitter.dataflow.map import Cut
from splitter.struct import IteratorVideoStream

#gets the boundaries of the clips
def clip_boundaries(start,end,size):
	start_points = list(range(start, end, size))
	tuples = [(s, min(s+(size-1),end)) for s in start_points]
	return tuples

#reconciles boundaries
def find_clip_boundaries(clip, boundaries):
	c_start,c_end = clip
	c_start = max(c_start, min([gstart for gstart,_ in boundaries]))
	c_end = min(c_end, max([gend for _,gend in boundaries]))

	start_clip = None
	end_clip = None

	for i, bound in enumerate(boundaries):
		start,end = bound

		if start <= c_start and\
		   end >= c_start:
		   start_clip = i

		if start <= c_end and\
		   end >= c_end:
		   end_clip = i

	return start_clip, end_clip

#clip mask
def get_clip_split_merge(clip, boundaries):
	c_start,c_end = clip
	start_clip, end_clip = find_clip_boundaries(clip, boundaries)

	rtn = []
	for clip_index in range(start_clip, end_clip+1):
		b_start, b_end = boundaries[clip_index]

		rtn.append((clip_index, (max(b_start,c_start), min(b_end,c_end)) ))

	return rtn

#takes in start end clips
def materialize_clip(clip, boundaries, streams):
	execution_plan = get_clip_split_merge(clip, boundaries)
	subiterators = []

	for crop in execution_plan:
		index, bounds = crop
		start = bounds[0]#streams[index][1]
		bounds2 = (bounds[0] - start, bounds[1] - start)
		subiterators.append(streams[index][Cut(*bounds2)])

	#v = VideoTransform()
	return IteratorVideoStream(itertools.chain(*subiterators))


def cut_header(header, start, end):
	start_index = max(0, start - header['start'])
	end_index = min(end - start, header['end'] - header['start'])

	if start_index == 0 and \
	   end_index == header['end'] - header['start']:
	   return header

	bounding_boxes = header['bounding_boxes'][start_index:end_index]

	label_set = set()
	for frame in bounding_boxes:
		for label, bb in frame:
			label_set.add(label)
	new_header = copy.deepcopy(header)
	new_header['start'] = start
	new_header['end'] = end
	new_header['label_set'] = list(label_set)
	new_header['bounding_boxes'] = bounding_boxes
	return new_header

def crop_overlap(crop, bb):
	''' Check if crop and box are overlapping
	'''
	if max(crop[0], crop[2]) >= min(bb[0], bb[2])  and  max(bb[0], bb[2]) >= min(crop[0], crop[2]):
		if max(crop[1], crop[3]) >= min(bb[1], bb[3]) and  max(bb[1], bb[3]) >= min(crop[1], crop[3]):
			return True
	else:
		return False

def crop_header(header, crop):
	label_set = set()
	bound_box_crop = []
	bounding_boxes = header['bounding_boxes']
	for frame in bounding_boxes:
		for label, bb in frame:
			overlap = crop_overlap(crop, bb)
			if overlap:
				label_set.add(label)
				bound_box_crop.append((label, bb))

	new_header = copy.deepcopy(header)
	new_header['label_set'] = list(label_set)
	new_header['bounding_boxes'] = bound_box_crop
	return new_header