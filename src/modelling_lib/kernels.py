from abc import abstractmethod

from equinox import AbstractVar, Module


class Kernel(Module):
    # All kernels should have a length scale and variance
    length_scale: AbstractVar[float]
    variance: AbstractVar[float]

    @abstractmethod
    def fourier_logprior(self):
        pass


class DummyKernel(Kernel):
    length_scale: float
    variance: float

    def __init__(self, length_scale: float, variance: float):
        self.length_scale = length_scale
        self.variance = variance

    def fourier_logprior(self) -> float:
        return 0
