import sys
import environ

from splitter.dataflow.agg import counts
from splitter.full_manager.condition import Condition
from splitter.full_manager.full_video_processing import CropSplitter
from splitter.optimizer.splitter import SplitterOptimizer
from splitter.tracking.background import FixedCameraBGFGSegmenter

from splitter.full_manager.full_manager import *
from splitter.simple_manager.manager import *
from splitter.tracking.contour import KeyPoints
from splitter.tracking.event import ActivityMetric, Filter
from splitter.utils.testing_utils import get_size
from experiments.environ import logrecord


def runFull(src, cleanUp=True, limit=-1, background_scale=1, optimizer=False):
    if cleanUp:
        if os.path.exists('/tmp/videos'):
            shutil.rmtree('/tmp/videos')

    manager = FullStorageManager(CustomTagger(FixedCameraBGFGSegmenter().segment, batch_size=30), CropSplitter(), '/tmp/videos')
    start = time.time()
    output = manager.put(src, 'test', parallel = False, args={'encoding': XVID, 'size': -1, 'sample': 1.0,
                         'offset': 0, 'limit': limit, 'batch_size': 30, 'background_scale': background_scale})
    end = time.time()

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
    logrecord('full', ({'size': limit, 'optimizer': optimizer, 'file': src, 'background_scale': background_scale,
                        'folder_size': get_size('/tmp/videos')}), 'get', str(result), 's', 'put', str(end - start), 's')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Enter filename as argv[1]")
        exit(1)
    filename = sys.argv[1]
    background_scale = [0.2, 0.4, 0.6, 0.8, 1]
    for bs in background_scale:
        runFull(filename, background_scale=bs, optimizer=False)
        runFull(filename, background_scale=bs, optimizer=True)
