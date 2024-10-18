class Channel:
    pass

class AWGChannel(Channel):
    def __init__(self):
        super().__init__()
        self.data = None

    def load_data(self, data):
        self.data = data

class AWGIQChannel(Channel):
    def __init__(self, local_oscillator_frequency: float, intermediate_frequency:  float):
        super().__init__()
        self.local_oscillator_frequency = local_oscillator_frequency
        self.intermediate_frequency = intermediate_frequency