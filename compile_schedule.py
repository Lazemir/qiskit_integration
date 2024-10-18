from qiskit.pulse.schedule import Schedule

from qiskit.pulse.instructions import Play, Delay, SetPhase, ShiftPhase, SetFrequency, ShiftFrequency

from qiskit.pulse import Instruction

from qiskit.pulse import SymbolicPulse, Waveform
from qiskit.pulse.library.pulse import Pulse

from qiskit.pulse.channels import PulseChannel

from channels import Channel as AWGChannel

from qiskit.pulse.transforms import pad

from typing import Mapping, Iterable

import numpy.typing as npt
import numpy as np

from collections import defaultdict


def get_envelope(pulse: Pulse):
    if isinstance(pulse, SymbolicPulse):
        return pulse.get_waveform().samples
    if isinstance(pulse, Waveform):
        return pulse.samples
    raise TypeError(f'Unexpected pulse type: {type(pulse)}')


def distribute_instructions(instructions: tuple[tuple[int, Instruction], ...]) -> Mapping[PulseChannel, Iterable[tuple[int, Instruction]]]:
    result = defaultdict(list)
    for start_time, instruction in instructions:
        if hasattr(instruction, 'channel'):
            result[instruction.channel].append((start_time, instruction))

    return result


class Oscillator:
    def __init__(self, sample_rate: float, frequency: float = 0, phase: float = 0) -> None:
        self.frequency = frequency
        self.phase = phase
        self.sample_rate = sample_rate

    def get_modulation(self, length: int):
        omega = 2 * np.pi * self.frequency
        time_list = np.arange(length) / self.sample_rate
        result = np.exp(-1j * (omega * time_list + self.phase))
        self.phase += omega * length / self.sample_rate
        self.phase %= 2 * np.pi
        return result


class ScheduleCompiler:
    def __init__(self, channels_mapping: Mapping[PulseChannel, AWGChannel], sample_rate: float):
        self.channels_mapping = channels_mapping
        self.sample_rate = sample_rate

    def compile_schedule(self, schedule: Schedule) -> Mapping[AWGChannel, npt.NDArray[complex]]:
        # Filling all gaps with Delay instructions
        schedule = pad(schedule)

        instructions = distribute_instructions(schedule.instructions)

        result = {}
        for virtual_channel, channel_instructions in instructions.items():
            awg_channel = self.channels_mapping[virtual_channel]
            if awg_channel in result:
                result[awg_channel] = result[awg_channel] + self._compile_for_channel(channel_instructions)
            else:
                result[awg_channel] = self._compile_for_channel(channel_instructions)

        return result


    def _compile_for_channel(self, channel_instructions: Iterable[tuple[int, Instruction]]) -> npt.NDArray[complex]:
        oscillator = Oscillator(self.sample_rate)
        result = []
        for start_time, instruction in channel_instructions:
            if isinstance(instruction, Play):
                pulse = instruction.pulse
                envelope = get_envelope(pulse)
                modulation = oscillator.get_modulation(start_time, pulse.duration)
                result.append(envelope * modulation)
            elif isinstance(instruction, Delay):
                result.append(np.zeros(instruction.duration))
            elif isinstance(instruction, SetPhase):
                oscillator.phase = instruction.phase
            elif isinstance(instruction, ShiftPhase):
                oscillator.phase += instruction.phase
            elif isinstance(instruction, SetFrequency):
                oscillator.frequency = instruction.frequency
            elif isinstance(instruction, ShiftFrequency):
                oscillator.frequency += instruction.frequency

        return np.concatenate(result)


