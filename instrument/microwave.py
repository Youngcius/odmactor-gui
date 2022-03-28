import time
from RsInstrument import RsInstrument


# RsInstrument('USB0::0x0AAD::0x0054::104174::INSTR', True, True)
class Microwave(RsInstrument):
    """
    Microwave class based on RsInstrument (vendor: R&S)
    """

    def __init__(self):
        super(Microwave, self).__init__('USB0::0x0AAD::0x0054::104174::INSTR', True, True)

    def set_frequency(self, freq):
        # super(Microwave, self).write_float('FREQUENCY', freq)
        print('MW frequency:', freq)

    def set_power(self, power):
        # super(Microwave, self).write_float('POW', power)
        print('MW power:', power)

    def run_given_time(self, duration):
        self.start()
        time.sleep(duration)
        self.stop()

    def start(self):
        # self.write_bool('OUTPUT:STATE', True)
        print('MW started')

    def stop(self):
        # self.write_bool('OUTPUT:STATE', False)
        print('MW stopped')

    def close(self):
        super(Microwave, self).close()
