import environ
import sys
from environ import *

from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.tracking.background import FixedCameraBGFGSegmenter
from splitter.optimizer.splitter import SplitterOptimizer

from splitter.struct import *
from splitter.utils import *
from splitter.dataflow.map import *
from splitter.full_manager.full_manager import *
from splitter.utils.testing_utils import *
from splitter.dataflow.agg import *
from splitter.tracking.contour import *
from splitter.tracking.event import *
from splitter.core import *
from splitter.simple_manager.manager import *

import cv2
import numpy as np
from tqdm import tqdm

# loads directly from the mp4 file
def runNaive(src, tot=1000, sel=0.1):
    cleanUp()

    c = VideoStream(src, limit=tot)
    sel = sel / 2
    left = Box(1600, 1600, 1700, 1800)
    middle = Box(1825, 1600, 1975, 1800)
    right = Box(2050, 1600, 2175, 1800)
    # left = Box(1600/3, 1600/3, 1700/3, 1800/3)
    # middle = Box(1825/3, 1600/3, 1975/3, 1800/3)
    # right = Box(2050/3, 1600/3, 2175/3, 1800/3)

    # left = Box(1600 / 2, 1600 / 2, 1700 / 2, 1800 / 2)
    # middle = Box(1825 / 2, 1600 / 2, 1975 / 2, 1800 / 2)
    # right = Box(2050 / 2, 1600 / 2, 2175 / 2, 1800 / 2)
    pipelines = c[GoodKeyPoints()][ActivityMetric('left', left)][
        ActivityMetric('middle', middle)][ActivityMetric('right', right)][
        Filter('left', [1], 1, delay=25)][
        Filter('middle', [1], 1, delay=25)][
        Filter('right', [1], 1, delay=25)]

    result = count(pipelines, ['left', 'middle', 'right'], stats=True)

    logrecord('naive', ({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')


# Simple storage manager with temporal filters
def runSimple(src, tot=1000, sel=0.1):
    cleanUp()

    manager = SimpleStorageManager('videos')
    now = timer()
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 20,
                      'num_processes': 4})
    put_time = timer() - now
    print("Put time for simple:", put_time)

    left = Box(1600, 1600, 1700, 1800)
    middle = Box(1825, 1600, 1975, 1800)
    right = Box(2050, 1600, 2175, 1800)

    sel = sel / 2

    clips = manager.get('test',
                        lambda f: overlap(f['start'], f['end'], tot // 2 - int(tot * sel), tot // 2 + int(tot * sel)))
    pipelines = []
    for c in clips:
        pipelines.append(c[GoodKeyPoints()][ActivityMetric('left', left)][
                             ActivityMetric('middle', middle)][ActivityMetric('right', right)][
                             Filter('left', [1], 1, delay=25)][
                             Filter('middle', [1], 1, delay=25)][
                             Filter('right', [1], 1, delay=25)])

    result = counts(pipelines, ['left', 'middle', 'right'], stats=True)

    logrecord('simple', ({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')


# Full storage manager with bg-fg optimization
def runFull(src, tot=1000, batch_size=20):
    # cleanUp()


    folder = '/bulk/videos'
    manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=batch_size), CropSplitter(),
                                 folder)
    # now = timer()
    # manager.put(src, 'test',
    #             args={'encoding': 'X264', 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': batch_size,
    #                   'num_processes': 4, 'background_scale': 1})
    # put_time = timer() - now
    # print("Put time for full:", put_time)
    print("Batch size:", batch_size, "Folder size:", get_size(folder))

    left = Box(1600, 1600, 1700, 1800)
    middle = Box(1825, 1600, 1975, 1800)
    right = Box(2050, 1600, 2175, 1800)

    # left = Box(1600 / 3, 1600 / 3, 1700 / 3, 1800 / 3)
    # middle = Box(1825 / 3, 1600 / 3, 1975 / 3, 1800 / 3)
    # right = Box(2050 / 3, 1600 / 3, 2175 / 3, 1800 / 3)

    # left = Box(1600 / 2, 1600 / 2, 1700 / 2, 1800 / 2)
    # middle = Box(1825 / 2, 1600 / 2, 1975 / 2, 1800 / 2)
    # right = Box(2050 / 2, 1600 / 2, 2175 / 2, 1800 / 2)


    clips = manager.get('test', Condition(label='foreground', custom_filter=None), large=True)
    pipelines = []

    total_counts = {}
    total_frames = 0
    total_time = 0

    for c in tqdm(clips):
        this_result = count(c[GoodKeyPoints()][ActivityMetric('left', left)][
                                ActivityMetric('middle', middle)][ActivityMetric('right', right)][
                                Filter('left', [1], 1, delay=25)][
                                Filter('middle', [1], 1, delay=25)][
                                Filter('right', [1], 1, delay=25)], ['left', 'middle', 'right'], stats=True)
        print(this_result)
        total_counts = {k: total_counts.get(k, 0) + this_result[0].get(k, 0) for k in set(total_counts) | set(this_result[0])}
        total_frames += this_result[1]['frames']
        total_time += this_result[1]['elapsed']

    result = total_counts, {'frames': total_frames, 'elapsed': total_time}

    logrecord('full', ({'size': tot, 'batch_size': batch_size, 'file': src, 'folder_size': get_size(folder)}), 'get', str(result), 's')


# All optimizations
def runFullOpt(src, tot=1000, sel=0.1):
    cleanUp()

    manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=20), CropSplitter(),
                                 'videos')
    now = timer()
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': 20,
                      'num_processes': 4, 'background_scale': 1})
    put_time = timer() - now
    print("Put time for full opt:", put_time)

    left = Box(1600, 1600, 1700, 1800)
    middle = Box(1825, 1600, 1975, 1800)
    right = Box(2050, 1600, 2175, 1800)
    sel = sel / 2

    clips = manager.get('test', Condition(label='foreground', custom_filter=None))

    pipelines = []
    d = SplitterOptimizer()
    for c in clips:
        pipeline = c[GoodKeyPoints()][ActivityMetric('left', left)][
            ActivityMetric('middle', middle)][ActivityMetric('right', right)][
            Filter('left', [1,1,1], 3, delay=25)][
            Filter('middle', [1,1,1], 3, delay=25)][
            Filter('right', [1,1,1], 3, delay=25)]
        pipeline = d.optimize(pipeline)
        pipelines.append(pipeline)

    result = counts(pipelines, ['left', 'middle', 'right'], stats=True)

    logrecord('fullopt', ({'size': tot, 'sel': sel, 'file': src}), 'get', str(result), 's')


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
#do_experiments(sys.argv[1], [runNaive, runSimple, runFull, runFullOpt], 600, range(9, 10))
do_experiments(sys.argv[1], [runFull], 600, range(9,10))
