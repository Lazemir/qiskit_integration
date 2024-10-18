from qiskit import pulse

from matplotlib import pyplot as plt

from compile_schedule import ScheduleCompiler
from channels import AWGChannel

def main() -> None:
    q1_01 = pulse.DriveChannel(0)
    q1_12 = pulse.DriveChannel(1)
    q2_01 = pulse.DriveChannel(2)

    awg_ch_1 = AWGChannel()
    awg_ch_2 = AWGChannel()

    channels_mapping = {
        q1_01: awg_ch_1,
        q1_12: awg_ch_1,
        q2_01: awg_ch_2,
    }

    x90 = pulse.Gaussian(200, 0.1, 20)

    with pulse.build() as pulse_prog:
        pulse.set_frequency(100e6, q1_01)
        pulse.set_frequency(200e6, q1_12)
        pulse.set_frequency(300e6, q2_01)
        pulse.play(x90, q1_01)
        pulse.play(x90, q1_12)
        pulse.play(x90, q2_01)

    pulse_prog = pulse.transforms.block_to_schedule(pulse_prog)

    pulse_prog.draw()
    plt.show()

    compiler = ScheduleCompiler(channels_mapping=channels_mapping, sample_rate=2e9)
    compiled_prog = compiler.compile_schedule(pulse_prog)

    for data in compiled_prog.values():
        plt.plot(data.real)
    plt.show()

if __name__ == '__main__':
    main()