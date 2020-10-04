"""This file is part of Splitter which is released under MIT License.

event.py defines some of the main detection primitives used in dlcv.
"""

import numpy as np

def moving_average(stream, k=10):
	buffer = np.zeros((len(stream)-k, 1))
	for i in range(k, len(stream)):
		buffer[i-k] = np.nanmedian(stream[i-k:i])

	return buffer

def moving_vol(stream, k=10):
	buffer = np.zeros((len(stream)-k, 1))
	for i in range(k, len(stream)):
		buffer[i-k] = np.std(stream[i-k:i])

	return buffer

def thresh_finder(stream, step=100, window=100, thresh=0.25):

	stream = moving_average(stream, window)

	buffer = []
	for i in range(step, len(stream), step):
		buffer.append(np.max(stream[i-step:i]))

	peaks = []

	for k in range(1,len(buffer)):
		if buffer[k] <= thresh:
			peaks.append(k*step + window)

	return peaks


def peak_finder(stream, step=100, window=100):

	stream = moving_average(stream, window)

	buffer = []
	for i in range(step, len(stream), step):
		buffer.append(np.nanmax(stream[i-step:i]))

	peaks = []

	for k in range(1,len(buffer)):

		print(buffer[k], buffer[k-1], buffer[k-2])

		if buffer[k-1] > buffer[k] and buffer[k-1] > buffer[k-2]:
			peaks.append(k*step + window)

	return peaks
