import numpy as np
import misc

class Tensor:
    def __init__(self, someIterable, dims: int | None = 0) -> None:
        try:
            self._tensor = np.array(someIterable, dtype = np.float64)
        except Exception as e:
            misc.printError("Initializing ns.Tensor", someIterable, "Please input a rectangular collection with all numbers")
            raise
        if not (isinstance(dims, int) and misc.nonnegative(dims)):
            misc.printError("Initializing ns.Tensor", dims, "Please provide zero or a positive integer for the dimensionality of the _tensor")
            raise Exception("")
        if dims != 0:
            if dims > self._tensor.ndim:
                while self._tensor.ndim < dims:
                    self._tensor = np.array([self._tensor])
            elif dims < self._tensor.ndim:
                while self._tensor.ndim > dims:
                    currShape = list(self._tensor.shape)
                    currShape[-2] = currShape[-2] * currShape[-1]
                    currShape = tuple(currShape[:len(currShape) - 1])
                    self._tensor = self._tensor.reshape(currShape)
    
    def __repr__(self) -> str:
        return str(self._tensor.tolist())
    
    def __str__(self) -> str:
        return str(self._tensor.tolist())
    
    def __getattr__(self, name):
        attrs = ["d", "T", "size", "shape"]
        if name == "d":
            return self._tensor.ndim
        elif name == "T":
            return self._tensor.T.tolist()
        elif name == "size":
            return self._tensor.size
        elif name == "shape":
            return self._tensor.shape
        else:
            misc.printError("Getting ns.Tensor attribute", name, f"Please enter a valid attribute for this tensor.\n\nHere are some valid attributes:\n{attrs}")
            raise Exception("")