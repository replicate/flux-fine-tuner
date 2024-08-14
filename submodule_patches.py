from torch.utils.data import DataLoader

"""
Getting this error when run without this patch.

File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/cog/server/worker.py", line 221, in _predict
result = predict(**payload)
File "/src/train.py", line 156, in train
job.run()
File "/src/ai-toolkit/jobs/ExtensionJob.py", line 22, in run
process.run()
File "/src/ai-toolkit/jobs/process/BaseSDTrainProcess.py", line 1595, in run
dataloader_iterator = iter(dataloader)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 440, in __iter__
return self._get_iterator()
File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 388, in _get_iterator
return _MultiProcessingDataLoaderIter(self)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1038, in __init__
w.start()
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/process.py", line 121, in start
self._popen = self._Popen(self)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/context.py", line 224, in _Popen
return _default_context.get_context().Process._Popen(process_obj)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/context.py", line 288, in _Popen
return Popen(process_obj)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/popen_spawn_posix.py", line 32, in __init__
super().__init__(process_obj)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/popen_fork.py", line 19, in __init__
self._launch(process_obj)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/popen_spawn_posix.py", line 47, in _launch
reduction.dump(process_obj, fp)
File "/root/.pyenv/versions/3.10.14/lib/python3.10/multiprocessing/reduction.py", line 60, in dump
ForkingPickler(file, protocol).dump(obj)
TypeError: cannot pickle 'weakref.ReferenceType' object
"""


def patch_dataloader():
    old_init = DataLoader.__init__

    def patched_init(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        if "prefetch_factor" in kwargs:
            del kwargs["prefetch_factor"]
        old_init(self, *args, **kwargs)

    DataLoader.__init__ = patched_init
