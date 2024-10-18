from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class Channel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def load_data(self, data: npt.NDArray[complex]):
        pass


class AWGChannel(Channel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

    def load_data(self, data: npt.NDArray[complex]):
        self.data = data.real


class AWGIQChannel(Channel):
    sample_rate = 2e9 # temporary

    def __init__(self, local_oscillator_frequency: float,
                 in_phase_channel: AWGChannel,
                 quadrature_channel: AWGChannel,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_oscillator_frequency = local_oscillator_frequency
        self.in_phase_channel = in_phase_channel
        self.quadrature_channel = quadrature_channel

    def _demodulate_data(self, data: npt.NDArray[complex]):
        omega = 2 * np.pi * self.local_oscillator_frequency
        time_list = np.arange(data.size) / self.sample_rate

        demodulation = np.exp(1j * omega * time_list)

        return data * demodulation

    def load_data(self, data: npt.NDArray[complex]):
        data = self._demodulate_data(data)
        self.in_phase_channel.load_data(data.real)
        self.quadrature_channel.load_data(data.imag)
