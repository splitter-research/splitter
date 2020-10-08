"""This file is part of Splitter which is released under MIT License.

constants.py defines encoding and numerical constants.
"""

#compression constants
GZ,BZ2,RAW = 'w:gz','w:bz2','w'

#encoding constants
XVID, DIVX, H264, MP4V, UNENC, GSC = 'XVID', 'DIVX', 'X264', 'FMP4', 'MJPG', 'Y800'

#ENCODINGS = [XVID, DIVX, H264, MP4V, UNENC, GSC]
ENCODINGS = [XVID]
FORMATS = [GZ,BZ2,RAW]
FOURCC = 828601953

#default file out
AVI = '.avi'
MKV = '.mkv'

#default frame rate
DEFAULT_FRAME_RATE = 30.0

#temp files
DEFAULT_TEMP = './tmp'

#language constructs
def TRUE(x):
	return True

def FALSE(x):
	return False

def hasLabel(l):
	return lambda x: l in x['label_set']

def startsBefore(time):
	return lambda x: time >= x['start']

def startsAfter(time):
	return lambda x: time < x['start']

def endsBefore(time):
	return lambda x: time >= x['end']

def endsAfter(time):
	return lambda x: time < x['end']

def AND(f1, f2):
	return lambda x: f1(x) and f2(x)

def OR(f1, f2):
	return lambda x: f1(x) or f2(x)
