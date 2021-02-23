#!/usr/bin/env python
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory as mpshm
import cloudpickle

try:
    mp.set_start_method("forkserver")
except RuntimeError:
    pass


def init_worker(shmmaps):
    for varname, spec in shmmaps.items():
        if varname == "FUNCTIONS":
            name = spec
            shm = mpshm.SharedMemory(name=name, create=False)
            globals()["FUNCTIONS"] = cloudpickle.loads(shm.buf)
        else:
            shape, dtype, name = spec
            shm = mpshm.SharedMemory(name=name, create=False)
            globals()["varname"] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)


def run_worker(task):
    func = FUNCTIONS[task["funcid"]]
    return func(*task["args"], **task["kwargs"])


class SharedMemory(object):
    def __init__(self, vars, nworkers):
        self._vars = vars
        self._nworkers = nworkers
        self._funcs = []
        self._tasks = []

    def __call__(self, progress=False):
        pass

    def __enter__(self):
        self._shms = {}
        self._shmmaps = {}
        for varname, arr in self._vars.items():
            self._shms[varname] = mpshm.SharedMemory(create=True, size=arr.nbytes)
            b = np.ndarray(arr.shape, dtype=arr.dtype, buffer=self._shms[varname].buf)
            b[:] = arr[:]
            self._shmmaps[varname] = (arr.shape, arr.dtype, self._shms[varname].name)

        return self

    def _func_shmap(self):
        payload = cloudpickle.dumps(self._funcs)
        self._shms["FUNCTIONS"] = mpshm.SharedMemory(create=True, size=len(payload))
        self._shms["FUNCTIONS"].buf[:] = payload
        self._shmmaps["FUNCTIONS"] = self._shms["FUNCTIONS"].name

    def __exit__(self, exc_type, exc_value, traceback):
        for varname, shm in self._shms.items():
            shm.close()
            shm.unlink()

    def register(self, func):
        funcid = len(self._funcs)
        self._funcs.append(func)

        def wrapper(*args, **kwargs):
            self._tasks.append({"args": args, "kwargs": kwargs, "funcid": funcid})

        return wrapper

    def evaluate(self, progress=False):
        if progress:
            import tqdm

        self._func_shmap()
        with mp.Pool(
            self._nworkers, initializer=init_worker, initargs=(self._shmmaps,)
        ) as p:
            if progress:
                result = list(
                    tqdm.tqdm(p.imap(run_worker, self._tasks), total=len(self._tasks))
                )
            else:
                result = p.map(run_worker, self._tasks)
        self._tasks = []
        return result
