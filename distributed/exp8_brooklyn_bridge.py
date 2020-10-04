import environ
import sys
from environ import *

from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import NullSplitter
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


def fixed_tagger(vstream, batch_size):
    count = 0
    for frame in vstream:
        count += 1
        if count >= batch_size:
            break
    if count == 0:
        raise StopIteration("Iterator is closed")
    return {'label': 'foreground', 'bb': Box(1600, 1600, 2175, 1800)}


# Full storage manager with bg-fg optimization
def runFull(src, tot=1000, batch_size=20):
    cleanUp()

    local_folder = '/var/www/html/videos'
    ip_addr = get_local_ip()
    remote_folder = 'http://' + ip_addr + '/videos'
    manager = FullStorageManager(CustomTagger(fixed_tagger, batch_size=batch_size), NullSplitter(),
                                 local_folder, remote_folder, dsn='dbname=header user=postgres password=splitter host=10.0.0.5')

    def put():
        now = timer()
        manager.put(src, 'test',
                    args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': tot, 'batch_size': batch_size,
                          'num_processes': 4, 'background_scale': 1})
        put_time = timer() - now
        print("Put time for full:", put_time)
        print("Batch size:", batch_size, "Folder size:", get_size(local_folder))

    def get():
        left = Box(1600, 1600, 1700, 1800)
        middle = Box(1825, 1600, 1975, 1800)
        right = Box(2050, 1600, 2175, 1800)

        # left = Box(1600 / 3, 1600 / 3, 1700 / 3, 1800 / 3)
        # middle = Box(1825 / 3, 1600 / 3, 1975 / 3, 1800 / 3)
        # right = Box(2050 / 3, 1600 / 3, 2175 / 3, 1800 / 3)

        # left = Box(1600 / 2, 1600 / 2, 1700 / 2, 1800 / 2)
        # middle = Box(1825 / 2, 1600 / 2, 1975 / 2, 1800 / 2)
        # right = Box(2050 / 2, 1600 / 2, 2175 / 2, 1800 / 2)


        clips = manager.get('test', Condition(label='foreground', custom_filter=None))
        pipelines = []

        now = timer()
        frame_count = 0
        for c in clips:
            for frame in c:
                frame_count += 1
            # pipelines.append(c[GoodKeyPoints()][ActivityMetric('left', left)][
            #                      ActivityMetric('middle', middle)][ActivityMetric('right', right)][
            #                      Filter('left', [1], 1, delay=25)][
            #                      Filter('middle', [1], 1, delay=25)][
            #                      Filter('right', [1], 1, delay=25)])

        # result = counts(pipelines, ['left', 'middle', 'right'], stats=True)
        result = timer() - now
        print(frame_count)

        logrecord('full', ({'size': tot, 'batch_size': batch_size, 'file': src, 'folder_size': get_size(local_folder)}), 'get', str(result), 's')

    put()
    get()


logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
#do_experiments(sys.argv[1], [runNaive, runSimple, runFull, runFullOpt], 600, range(9, 10))
do_experiments_batch_size(sys.argv[1], [runFull], -1, [72])