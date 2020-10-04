import sys
import environ

from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.tracking.background import FixedCameraBGFGSegmenter
from splitter.optimizer.splitter import SplitterOptimizer
from splitter.utils.testing_utils import get_size
from experiments.environ import logrecord

from splitter.full_manager.full_manager import *
from splitter.dataflow.agg import *
from splitter.tracking.contour import *
from splitter.tracking.event import *
from splitter.simple_manager.manager import *


def runFull(src, cache=False, cleanUp=True, limit=6000, optimizer=True):
    if cleanUp:
        if os.path.exists('/tmp/videos'):
            shutil.rmtree('/tmp/videos')

    manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=30), CropSplitter(),
                                 '/tmp/videos')
    manager.put(src, 'test',
                args={'encoding': XVID, 'size': -1, 'sample': 1.0, 'offset': 0, 'limit': limit, 'batch_size': 30,
                      'num_processes': 4, 'background_scale': 1}, hwang=False)
    if cache:
        manager.cache('test', Condition(label='foreground'), hwang=False)

    clips = manager.get('test', Condition(label='foreground'))

    region = Box(200, 550, 350, 750)

    pipelines = []
    d = SplitterOptimizer()
    for c in clips:
        pipeline = c[KeyPoints()][ActivityMetric('one', region)][
            Filter('one', [-0.25, -0.25, 1, -0.25, -0.25], 1.5, delay=10)]
        if optimizer:
            pipeline = d.optimize(pipeline)
        pipelines.append(pipeline)

    result = counts(pipelines, ['one'], stats=True)
    logrecord('full', ({'size': limit, 'cache': cache, 'optimizer': optimizer, 'file': src,
                         'folder_size': get_size('/tmp/videos')}), 'get', str(result), 's')
    if cache:
        manager.uncache('test', Condition(label='foreground'))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Enter filename as argv[1]")
        exit(1)
    filename = sys.argv[1]
    limit_list = [150, 300, 600, 1200, 2400, 4800]
    for limit in limit_list:
        runFull(filename, cache=False, limit=limit, optimizer=False)
        runFull(filename, cache=False, limit=limit, optimizer=True)
        runFull(filename, cache=True, limit=limit, optimizer=False)
        runFull(filename, cache=True, limit=limit, optimizer=True)