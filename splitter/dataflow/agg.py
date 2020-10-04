"""This file is part of Splitter which is released under MIT License.

agg.py defines aggregation functions
"""

from splitter.dataflow.validation import check_metrics_and_filters, countable
from splitter.struct import IteratorVideoStream
from splitter.dataflow.xform import Null

import logging
import time
import itertools

def count(stream, keys, stats=False):
	"""Count counts the true hits of a defined event.
	"""

	#actual logic is here
	counter = {}
	frame_count = 0
	now = time.time()
	for frame in stream:
		frame_count += 1

		if frame_count == 1:
			logging.info("Processing first frame of stream")

		for key in keys:
			if frame[key]:
				subkey = key + '_' + str(frame[key])
				counter[subkey] = counter.get(subkey,0) + 1

	# profiling
	for obj in stream.lineage():
		if hasattr(obj, "time_elapsed"):
			logging.info("%s: %s" % (type(obj).__name__, obj.time_elapsed))
		else:
			logging.info("%s time not measured" % type(obj).__name__)

	if not stats:
		return counter
	else:
		return counter, {'frames': frame_count, \
						 'elapsed': (time.time() - now)}

def counts(streams, keys, stats=False):
	"""Count counts the true hits of a defined event.
	"""
	stream = IteratorVideoStream(itertools.chain(*streams), streams)

	lineage = []
	for s in streams:
		lineage.extend(s.lineage())

	stream.global_lineage = lineage

	return count(stream, keys, stats)


def get(stream, key, frame_rate=-1):
	if frame_rate == -1:
		return [(v['frame'], v['data']) for v in stream if v[key]]
	else:
		return [( int(v['frame']/frame_rate) , v['data']) for v in stream if v[key]]
