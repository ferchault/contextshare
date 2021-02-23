contextshare
============

Simple interface for numpy arrays in multiprocessing contexts.

Example
-------

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

Credits
-------

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) using the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage)` template.
