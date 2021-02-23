contextshare
============

Simple context manager to share numpy arrays in python multiprocessing.

Example
-------

The following code example is complete and compares a `serial()` implementation  with the `parallel()` version thereof implemented with `contextshare`.

```python
#!/usr/bin/env python
import contextshare as shared
import numpy as np


def serial(sigmas):
    alphas = []
    for sigma in sigmas:
        K = np.exp(-0.5 * D / (sigma * sigma))
        alphas.append(np.linalg.solve(K, b))
    return alphas


def parallel(sigmas):
    with shared.SharedMemory({"D": D, "b": b}, nworkers=11) as sm:

        @sm.register
        def one_sigma(sigma):
            K = np.exp(-0.5 * D / (sigma * sigma))
            return np.linalg.solve(K, b)

        for sigma in sigmas:
            one_sigma(sigma)

        return sm.evaluate(progress=True)


if __name__ == "__main__":
    N = 2000
    D = np.random.random((N, N))
    b = np.random.random(N)

    sigmas = 2.0 ** np.arange(1, 12)

    s = serial(sigmas)
    p = parallel(sigmas)
   
```

Documentation
-------------
The context manager `shared.SharedMemory` takes two arguments: the dictionary of numpy arrays to make available on all parallel workers and the number of workers to create.

```python
# makes variables D and b available under the same name on 11 workers
with shared.SharedMemory({"D": D, "b": b}, nworkers=11) as sm:
```

The workers are spawned upon entering the context and are stopped upon exiting. Shared memory references are cleaned up automatically. Note that only numpy arrays are supported for sharing. Other arguments should be placed in the arguments of the function below. The function to call (i.e. the body of the serial for loop) needs to be placed in a function and either decorated or called explicitly:

```python
# decorator
@sm.register
def one_sigma(sigma):
    pass

# or, equivalently, an explicit call
sm.register(one_sigma)        
```
Calling this decorated function returns immediately and enqueues a function call. Calling 

```python
sm.evaluate(progress=True)
```
starts the calculations and returns the results in order. With `progress=True` a progress bar is shown, default is to be silent.



Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) using the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)` template.
