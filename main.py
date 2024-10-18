from qiskit import pulse

from matplotlib import pyplot as plt

from compile_schedule import ScheduleCompiler
from channels import AWGChannel, AWGIQChannel


def main() -> None:
    q1_01 = pulse.DriveChannel(0)
    q1_12 = pulse.DriveChannel(1)
    q2_01 = pulse.DriveChannel(2)
    readout_q1 = pulse.MeasureChannel(0)
    readout_q2 = pulse.MeasureChannel(1)

    awg_ch_q1 = AWGChannel(name='q1_low_frequency')
    awg_iq_ch_q1 = AWGIQChannel(name='q1_high_frequency',
                                local_oscillator_frequency=5e9,
                                in_phase_channel=AWGChannel('q1_in_phase'),
                                quadrature_channel=AWGChannel('q1_quadrature'))
    awg_ch_q2 = AWGChannel(name='q2_low_frequency')
    awg_iq_readout = AWGIQChannel(name='readout',
                                  local_oscillator_frequency=7e9,
                                  in_phase_channel=AWGChannel('readout_in_phase'),
                                  quadrature_channel=AWGChannel('readout_quadrature'))

    channels_mapping = {
        q1_01: awg_ch_q1,
        q1_12: awg_iq_ch_q1,
        q2_01: awg_ch_q2,
        readout_q1: awg_iq_readout,
        readout_q2: awg_iq_readout,
    }

    x90 = pulse.Constant(500, 1)
    m = pulse.Gaussian(1000, 0.5, sigma=500)

    with pulse.build() as pulse_prog:
        pulse.set_frequency(700e6, q1_01)
        pulse.set_frequency(5e9, q1_12)

        pulse.set_frequency(750e6, q2_01)

        pulse.set_frequency(6.8e9, readout_q1)
        pulse.set_frequency(7.1e9, readout_q2)

        with pulse.align_sequential():
            with pulse.align_left():
                pulse.play(x90, q1_01)
                pulse.play(x90, q2_01)

            pulse.play(x90, q1_12)
            pulse.play(x90, q1_01)

            with pulse.align_left():
                pulse.play(m, readout_q1)
                pulse.play(m, readout_q2)

    pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

    pulse_prog.draw()
    plt.show()

    compiler = ScheduleCompiler(channels_mapping=channels_mapping, sample_rate=2e9)
    compiled_prog = compiler.compile_schedule(pulse_prog)

    for awg_channel, data in compiled_prog.items():
        awg_channel.load_data(data)

    for awg_channel in compiled_prog.keys():
        if isinstance(awg_channel, AWGChannel):
            plt.plot(awg_channel.data, label=awg_channel.name)
        elif isinstance(awg_channel, AWGIQChannel):
            for ch in (awg_channel.in_phase_channel, awg_channel.quadrature_channel):
                plt.plot(ch.data, label=ch.name)
        else:
            raise TypeError('Unexpected type')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()