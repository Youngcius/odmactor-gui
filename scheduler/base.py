"""
Scheduler abstract base class
"""

import abc
import datetime
import time
import json
import os
import pickle
import copy
import nidaqmx
import numpy as np
import scipy.constants as C
import TimeTagger as tt
from tqdm import tqdm
from instrument import ASG, Microwave, LockInAmplifier, Laser
from utils import dBm_to_mW, mW_to_dBm
from utils.sequence import expand_to_same_length
from typing import List, Any, Optional, Union
from utils.sequence import sequences_to_string, sequences_to_figure
from matplotlib.figure import Figure


class Scheduler(abc.ABC):
    """
    ODMR manipulation scheduler base class
    """

    def __init__(self, *args, **kwargs):
        self._cache: Any = None
        self._data = []
        self._data_ref = []
        self.name = 'Base Scheduler'
        # pi pulse, for spin manipulation
        self.pi_pulse = {'freq': None, 'power': None, 'time': None}  # unit: Hz, dBm, s
        self._result = []
        self._result_detail = {}
        self._freqs = []  # unit: Hz
        self._times = []  # unit: ns
        self._cur_freq = 1e9
        self._cur_time = 0
        self._asg_sequences = []
        self.reset_asg_sequence()
        self._asg_conf = {'t': 0, 'N': 0}  # to calculate self.asg_dwell = N * t
        self._mw_conf = {'freq': C.giga, 'power': 0}  # current MW parameter settings
        self._configuration = {}
        self._laser_control = True
        self.is_running = False
        self.two_pulse_readout = False  # whether to use double-pulse readout

        # connect to Laser, MW, ASG, Tagger/Lockin

        self.mw_exec_mode = ''
        self.mw_exec_modes_optional = {'scan-center-span', 'scan-start-stop'}
        self.channel = {'laser': 1, 'mw': 2, 'apd': 3, 'tagger': 5, 'mw_sync': 4, 'lockin_sync': 8}
        self.tagger_input = {'apd': 1, 'asg': 2}
        self.daqtask: nidaqmx.Task = None
        self.counter: tt.IteratorBase = None

        # properties or method for debugging
        self.sync_delay = 0.0
        self.mw_dwell = 0.0
        self.asg_dwell = 0.0
        self.time_pad = 0.0
        self.time_total = 0.0  # total time for scanning frequencies (estimated)
        default_output_dir = os.path.join(os.path.expanduser('~'), 'Downloads', 'output-' + str(datetime.date.today()))
        self.output_dir = kwargs.get('output_dir', default_output_dir)
        # self.output_dir = '../output/'
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # 1: high-level effective; 0: low level effective
        self.laser_ttl = kwargs.get('laser_ttl', 1)
        self.mw_ttl = kwargs.get('mw_ttl', 1)
        self.apd_ttl = kwargs.get('apd_ttl', 1)
        self.tagger_ttl = kwargs.get('tagger_ttl', 1)

        self.with_ref = kwargs.get('with_ref', True)

        self.epoch_omit = kwargs.get('epoch_omit', 0)

        # synchronization frequency between MW and Lockin, unit: Hz
        self.sync_freq = kwargs.get('sync_freq', 50)

        # high-order dynamical decoupling order
        self.order = kwargs.get('order', 1)

        # on/off MW when on/off ASG's MW channel
        self.mw_on_off = kwargs.get('mw_on_off', False)

        # set False when MW switch controlled  by ASG is no available
        self.asg_control_mw_on_off = kwargs.get('asg_control_mw_on_off', True)

        # use lockin or tagger
        self.use_lockin = kwargs.get('use_lockin', False)

        # output lock-in sync sequence from ASG or not
        self.output_lockin = kwargs.get('output_lockin', False)

        # configure instruments from external variables
        self.laser = kwargs.get('laser')  # if not designated, it is None
        self.asg = kwargs.get('asg')
        self.mw = kwargs.get('mw')
        self.tagger = kwargs.get('tagger')
        self.lockin = kwargs.get('lockin')

        # initialize instruments
        if self.laser is None:
            self.laser = Laser()
        if self.asg is None:
            self.asg = ASG()
        if self.mw is None:
            try:
                self.mw = Microwave()
            except:
                self.mw = None
        if self.use_lockin:
            if self.lockin is None:
                try:
                    self.lockin = LockInAmplifier()
                except:
                    self.lockin = None
        else:
            if self.tagger is None and not tt.scanTimeTagger():
                self.tagger = tt.createTimeTagger()

    def print_info(self):
        print(self.name)
        print(f'laser: {self.laser}, asg: {self.asg}, mw: {self.mw}, tagger: {self.tagger}, lockin: {self.lockin}')
        print()

    def connect(self):
        """
        Check and connect all instruments
        ---
        Laser, MW, ADG, Tagger, Lockin
        """
        self.laser.connect()
        self.asg.connect()

        if self.mw is not None:
            self.mw.connect()

        if self.use_lockin:
            self.lockin = LockInAmplifier()
        else:
            try:
                self.tagger.getSerial()
            except:
                self.tagger = tt.createTimeTagger()

    def set_asg_sequences_ttl(self, laser_ttl=None, mw_ttl=None, apd_ttl=None, tagger_ttl=None):
        """
        Set the ASG sequences TTL high-effective or low-effective
        1 means high-effective, 0 meas low-effective
        :param laser_ttl: TTL effectiveness of Laser channel sequence
        :param mw_ttl:  TTL effectiveness of MW channel sequence
        :param apd_ttl: TTL effectiveness of APD channel sequence
        :param tagger_ttl: TTL effectiveness of Tagger sequence
        """
        if laser_ttl is not None:
            self.laser_ttl = laser_ttl
        if mw_ttl is not None:
            self.mw_ttl = mw_ttl
        if apd_ttl is not None:
            self.apd_ttl = apd_ttl
        if tagger_ttl is not None:
            self.tagger_ttl = tagger_ttl

    def download_asg_sequences(self, laser_seq: List[int] = None, mw_seq: List[int] = None,
                               tagger_seq: List[int] = None, sync_seq: List[int] = None):
        """
        Download control sequences into the memory of ASG
        :param laser_seq: laser control sequence
        :param mw_seq: MW control sequence
        :param tagger_seq: tagger readout control sequence
        :param sync_seq: synchronization sequence between Lock-in amplifier and MW
        """
        sequences = [laser_seq, mw_seq, tagger_seq]
        if not any(sequences):
            raise ValueError('laser_seq, mw_seq and tagger_seq cannot be all None')
        sequences = [seq for seq in sequences if seq is not None and sum(seq) > 0]  # non-None sequences
        lengths = np.unique(list(map(sum, sequences)))
        if len(lengths) != 1:
            raise ValueError('laser/mw/tagger sequences should have the same length')

        # configure ASG period information
        self.reset_asg_sequence()
        if laser_seq is not None:
            self._asg_sequences[self.channel['laser'] - 1] = laser_seq
        if mw_seq is not None:
            self._asg_sequences[self.channel['mw'] - 1] = mw_seq
        if tagger_seq is not None:
            self._asg_sequences[self.channel['tagger'] - 1] = tagger_seq
        if sync_seq is not None:
            self._asg_sequences[self.channel['mw_sync'] - 1] = sync_seq
            self._asg_sequences[self.channel['lockin_sync'] - 1] = sync_seq

        # connect & download pulse data
        if self.output_lockin:
            self.asg.load_data(self._asg_sequences)
        else:
            seqs = copy.deepcopy(self._asg_sequences)
            seqs[self.channel['mw_sync'] - 1], seqs[self.channel['lockin_sync'] - 1] = [0, 0], [0, 0]
            self.asg.load_data(seqs)

    def configure_lockin_counting(self, channel: str = 'Dev1/ai0', freq: int = None):
        """
        Configure counter building on Lock-in Amplifier and NI DAQ
        :param channel: output channel from NIDAQ to PC
        :param freq: synchronization frequency between MW and Lockin
        """
        self.daqtask = nidaqmx.Task()
        self.daqtask.ai_channels.add_ai_voltage_chan(channel)
        if freq is not None:
            self.sync_freq = freq

    def configure_tagger_counting(self, apd_channel: int = None, asg_channel: int = None, reader: str = 'counter'):
        """
        Configure counter building on APD and Time Tagger
        Configure asg-channel and apd-channel for ASG. For Swabian Time Tagger, channel number range: [1, 8].
        :param apd_channel: APD channel number
        :param asg_channel: ASG channel number
        :param reader: counter of specific readout type
        """
        self.two_pulse_readout = False
        if apd_channel is not None:
            self.tagger_input['apd'] = apd_channel
        if asg_channel is not None:
            self.tagger_input['asg'] = asg_channel
        print('Current Tagger input channels:', self.tagger_input)

        # construct & execute Measurement instance
        N = self._asg_conf['N']

        if reader == 'counter':
            # continuous counting
            t_ps = int(self._asg_conf['t'] / C.pico)
            self.counter = tt.Counter(self.tagger, channels=[self.tagger_input['apd']], binwidth=t_ps, n_values=N)
        elif reader == 'cbm':
            # pulse readout
            if self.two_pulse_readout:
                self.counter = tt.CountBetweenMarkers(self.tagger, self.tagger_input['apd'],
                                                      begin_channel=self.tagger_input['asg'],
                                                      end_channel=-self.tagger_input['asg'],
                                                      n_values=N * 2)
            else:
                self.counter = tt.CountBetweenMarkers(self.tagger, self.tagger_input['apd'],
                                                      begin_channel=self.tagger_input['asg'],
                                                      end_channel=self.tagger_input['asg'], n_values=N)
        else:
            raise ValueError('unsupported reader (counter) type')

    def _start_device(self):
        """
        Start device: MW, ASG; Execute Measurement instance.
        """
        # 1. run ASG firstly
        self._data.clear()
        self._data_ref.clear()
        self.asg.start()

        # 2. restart self.counter if using Tagger
        if not self.use_lockin:
            self.counter.start()

        # 3. run MW then
        self.mw.start()
        # print('MW on/off status:', self.mw.instrument_status_checking)

    def _acquire_data_to_cache(self, cache):
        """
        Acquire data and save it to a buffer region
        :param cache: data cache, e.g., a list instance
        """
        if self.use_lockin:
            time.sleep(self.time_pad)
            time.sleep(self.asg_dwell)
            cache.append(self.daqtask.read(number_of_samples_per_channel=1000))
        else:
            # from tagger
            self.counter.clear()
            time.sleep(self.time_pad)
            time.sleep(self.asg_dwell)
            cache.append(self.counter.getData().ravel().tolist())

    def _get_data(self):
        """
        Read signal data from data acquisition devices, i.e., APD + Tagger, or Lock-in + DAQ
        ---
        1. with Time Tagger
            read one value in each ASG operation period, totally N values
        2. with Lock-in Amplifier
            read M values after the last ASG operation period, M is not necessarily equal to N
        """
        self._acquire_data_to_cache(self._data)

    def _get_data_ref(self):
        """
        Read reference data from data acquisition devices, i.e., APD + Tagger, or Lock-in + DAQ
        ---
        1. with Time Tagger
            read one value in each ASG operation period, totally N values
        2. with Lock-in Amplifier
            read M values after the last ASG operation period, M is not necessarily equal to N
        """
        self._acquire_data_to_cache(self._data_ref)

    def run(self):
        """
        A rough scheduling method
        1) start device
        2) acquire data timely
        """
        self._start_device()
        self._acquire_data()
        self.stop()

    def stop(self):
        """
        Stop hardware (ASG, MW, Tagger) scheduling
        """
        if self.counter is not None:
            self.counter.stop()
        if self.daqtask is not None:
            self.daqtask.stop()
        self.asg.stop()
        self.mw.stop()
        print('Stopped: Scheduling process has stopped')

    def close(self):
        """
        Release instrument (ASG, MW, Tagger) resources
        """
        if self.asg is not None:
            self.asg.close()
        if self.mw is not None:
            self.mw.close()
        if not self.use_lockin and self.tagger is not None:
            tt.freeTimeTagger(self.tagger)
        if self.use_lockin and self.daqtask is not None:
            self.daqtask.close()
        print('Closed: All instrument resources has been released')

    def configure_mw_paras(self, power: float = None, freq: float = None, regulate_pi: bool = False, *args, **kwargs):
        """
        Configure parameters of MW instrument
        :param power: power of MW, unit: dBm
        :param freq: frequency of MW, unit: Hz
        :param regulate_pi: whether regulate built-int MW pi pulse of this Scheduler
        """
        if power is not None:
            self._mw_conf['power'] = power
            self.mw.set_power(power)
            if regulate_pi:  # regulate time duration based on MW power
                self._regulate_pi_pulse(power=power)  # by power
        if freq is not None:
            self._mw_conf['freq'] = freq
            self.mw.set_frequency(freq)

        self.mw.start()

    def _regulate_pi_pulse(self, power: float = None, time: float = None):
        """
        Regulate time duration of MW pi pulse according to designed MW power, or vice verse
        :param power: MW power, unit: dBm
        :param time: time duration of MW pi pulse, unit: s
        :return:
        """
        if not any(self.pi_pulse.values()):  # means it has no pi-pulse parameters
            raise TypeError('No pi pulse parameters to be regulated')
        else:
            power_ori = self.pi_pulse['power']  # unit: dBm
            time_ori = self.pi_pulse['time']
            if power is not None:
                # calculate new "time"
                time = np.sqrt(10 ** ((power_ori - power) / 10)) * time_ori
            elif time is not None:
                # calculate new "power"
                power = mW_to_dBm((time_ori / time) ** 2 * dBm_to_mW(power_ori))
            self.pi_pulse['power'] = power
            self.pi_pulse['time'] = time

    def reset_asg_sequence(self):
        """
        Reset all channels of ASG as ZERO signals
        """
        self._asg_sequences = [[0, 0] for _ in range(8)]

    def save_configuration(self, fname: str = None):
        """
        Save configuration parameters into a disk file
        :param fname: file name for saving
        """
        if fname is None:
            fname = '{} {}'.format(self.name, datetime.date.today())
        with open(fname, 'wr') as f:
            pickle.dump(self, f)
        print('Scheduler configuration has been save to {}'.format(fname))

    def laser_on_seq(self):
        """
        Set sequence to control Laser keeping on during the whole period
        """
        idx_laser_channel = self.channel['laser'] - 1
        t = sum(self._asg_sequences[idx_laser_channel])
        self._asg_sequences[idx_laser_channel] = [t, 0]
        self.asg.load_data(self._asg_sequences)
        self.asg.start()

    def laser_off_seq(self):
        """
        Set sequence to control Laser keeping off during the whole period
        """
        idx_laser_channel = self.channel['laser'] - 1
        self._asg_sequences[idx_laser_channel] = [0, 0]
        self.asg.load_data(self._asg_sequences)
        self.asg.start()

    def mw_on_seq(self):
        """
        Set sequence to control MW keeping on during the whole period
        """
        idx_mw_channel = self.channel['mw'] - 1
        t = sum(self._asg_sequences[idx_mw_channel])
        if self.mw_ttl == 0:
            mw_seq = [0, t]
        else:
            mw_seq = [t, 0]
        self._asg_sequences[idx_mw_channel] = mw_seq
        self.asg.load_data(self._asg_sequences)
        self.asg.start()

    def mw_off_seq(self):
        """
        Set sequence to control MW keeping off during the whole period
        """
        mw_seq = [0, 0]
        idx_mw_channel = self.channel['mw'] - 1
        self._asg_sequences[idx_mw_channel] = mw_seq
        self.asg.load_data(self._asg_sequences)
        self.asg.start()

    def mw_control_seq(self, mw_seq: List[int] = None) -> Optional[List[int]]:
        """
        Get or set current MW control sequence
        :param mw_seq: designed MW control sequence, optional parameter
        :return: return current MW control sequence when mw_seq designed, otherwise return None
        """
        idx_mw_channel = self.channel['mw'] - 1
        if mw_seq is None:
            return self._asg_sequences[idx_mw_channel]
        else:
            self._asg_sequences[idx_mw_channel] = mw_seq
            self.asg.load_data(self._asg_sequences)
            self.asg.start()

    def _conf_time_paras(self, t, N=10000):
        """
        Configure characteristic time parameters
        :param t: total time of one ASG sequence period, unit: ns
        :param N: repetition number of ASG sequences periods for each detection point
        """
        self._asg_conf['t'] = t * C.nano  # unit: s
        self._asg_conf['N'] = N
        self.asg_dwell = self._asg_conf['N'] * self._asg_conf['t']  # duration without padding
        self.time_pad = 0.01 * self.asg_dwell  # 0.035

    def _cal_counts_result(self):
        """
        Calculate counts of data, with respect to frequencies (scanning mode)
        If using two-pulse-readout strategy, calculate both signals and reference signals
        """
        if isinstance(self, FrequencyDomainScheduler):
            xs_name = 'freqs'
            xs = self._freqs
        elif isinstance(self, TimeDomainScheduler):
            xs_name = 'times'
            xs = self._times
        else:
            raise TypeError('unsupported function in this scheduler type')

        if self.with_ref:
            counts = [np.mean(ls) for ls in self._data]
            counts_ref = [np.mean(ls) for ls in self._data_ref]
            self._result = [xs, counts, counts_ref]
            self._result_detail = {
                xs_name: xs,
                'counts': counts,
                'counts_ref': counts_ref,
                'origin_data': self._data,
                'origin_data_ref': self._data_ref
            }
        else:
            if self.two_pulse_readout:
                counts_pairs = [(np.mean(ls[1::2]), np.mean(ls[::2])) for ls in self._data]
                counts = list(map(min, counts_pairs))
                counts_ref = list(map(max, counts_pairs))
                self._result = [xs, counts, counts_ref]
                self._result_detail = {
                    xs_name: xs,
                    'counts': counts,
                    'counts_ref': counts_ref,
                    'origin_data': self._data,
                }
            else:
                counts = [np.mean(ls) for ls in self._data]
                self._result = [xs, counts]
                self._result_detail = {
                    xs_name: xs,
                    'counts': counts,
                    'origin_data': self._data
                }

    def _gene_data_result_fname(self, fmt: str = None) -> str:
        """
        Generate file name of data acquisition result, based on time, data and random numbers
        :return: file name, str type
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        sub_dir = os.path.join(self.output_dir, str(datetime.date.today()))
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        if self.with_ref:
            # calculate signals and reference counts
            fname = os.path.join(sub_dir, '{}-counts-with-ref-{}'.format(self.name.split()[0], timestamp))
        else:
            # just calculate signal counts
            fname = os.path.join(sub_dir, '{}-counts-{}'.format(self.name.split()[0], timestamp))
        if fmt is not None:
            fname = fname + '.{}'.format(fmt)
        return fname

    def save_result(self, fname: str = None):
        """
        Save self.result_detail property
        :param fname: if not designed, will be randomly generated
        """
        if not self._result or not self._result_detail:
            raise ValueError('empty result cannot be saved')
        if fname is None:
            fname = self._gene_data_result_fname('json')
        with open(fname + '.json', 'w') as f:
            json.dump(self._result_detail, f)
        self.output_fname = fname + '.json'
        print('Detailed data result has been saved into {}'.format(fname + '.json'))

    def __str__(self):
        return self.name

    def config_sequences(self, sequences: List[List[Union[float, int]]]):
        """
        Configure ODMR sequences from external sequences data directly
        """
        self._asg_sequences = sequences
        self.asg.load_data(sequences)

    @property
    def sequences(self):
        return self._asg_sequences

    @property
    def sequences_no_sync(self):
        seqs = copy.deepcopy(self._asg_sequences)
        seqs[self.channel['mw_sync'] - 1], seqs[self.channel['lockin_sync'] - 1] = [0, 0], [0, 0]
        return seqs

    @property
    def frequencies(self):
        return self._freqs

    @property
    def times(self):
        return self._times

    @property
    def cur_freq(self):
        return self._cur_freq

    @property
    def cur_time(self):
        return self._cur_time

    @property
    def cur_data(self, aggregate='mean'):
        if aggregate == 'mean':
            return [np.mean(ls) for ls in self._data]
        elif aggregate == 'sum':
            return [np.sum(ls) for ls in self._data]
        else:
            raise ValueError('Not supported aggregation type')

    @property
    def cur_data_ref(self, aggregate='mean'):
        if aggregate == 'mean':
            return [np.mean(ls) for ls in self._data_ref]
        elif aggregate == 'sum':
            return [np.sum(ls) for ls in self._data_ref]
        else:
            raise ValueError('Not supported aggregation type')

    @abc.abstractmethod
    def configure_odmr_seq(self, *args, **kwargs):
        """
        Configure ODMR detection sequences parameters
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _acquire_data(self, *args, **kwargs):
        """
        Acquire data in real time
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_single_step(self, *args, **kwargs):
        """
        Run the Scheduler at one single detection parameter setting point
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run_scanning(self, *args, **kwargs):
        """
        Scanning mode to run this Scheduler
        """
        raise NotImplementedError

    @property
    def laser_control(self) -> bool:
        """
        True: laser-controlled measurement
        False: pure spin manipulation
        """
        return self._laser_control

    @property
    def result(self) -> List[List[float]]:
        return self._result

    @property
    def result_detail(self) -> dict:
        return self._result_detail

    @property
    def sequences_strings(self) -> str:
        return sequences_to_string(self._asg_sequences)

    @property
    def sequences_figure(self) -> Figure:
        """
        Ignore the lock-in frequency synchronization channel outputted to MW and Lock-in Amplifier
        """
        seqs = copy.deepcopy(self._asg_sequences)
        seqs[self.channel['mw_sync'] - 1], seqs[self.channel['lockin_sync'] - 1] = [0, 0], [0, 0]
        return sequences_to_figure(seqs)


class FrequencyDomainScheduler(Scheduler):
    """
    Frequency-domain ODMR detection abstract class
    """

    def __init__(self, *args, **kwargs):
        super(FrequencyDomainScheduler, self).__init__(*args, **kwargs)
        self.name = 'Base ODMR Scheduler'

    def set_mw_freqs(self, start, end, step):
        """
        Set frequencies for scanning detection (e.g. CW-ODMR, Pulse-ODMR)
        All unit is "Hz"
        :param start: start frequency
        :param end: end frequency
        :param step: frequency step
        """
        # unit: Hz
        # n_freqs = int((end - start) / step + 1)
        self._freqs = np.arange(start, end + step / 2, step).tolist()
        n_freqs = len(self._freqs)
        if self.asg_dwell == 0:
            raise ValueError('"asg_dwell" is 0.0 currently. Please set ODMR sequences parameters firstly.')
        else:
            self.time_total = self.asg_dwell * n_freqs * 2 if self.with_ref else self.asg_dwell * n_freqs

    def _scan_freqs_and_get_data(self):
        """
        Scanning frequencies & getting data of Counter
        """
        # omit several scanning points
        for _ in range(self.epoch_omit):
            self.mw.set_frequency(self._freqs[0])
            time.sleep(self.time_pad + self.asg_dwell)
            if self.with_ref:
                time.sleep(self.time_pad + self.asg_dwell)

        # formal data acquisition
        mw_on_seq = self._asg_sequences[self.channel['mw'] - 1]
        for freq in tqdm(self._freqs):
            self._cur_freq = freq
            self.mw.set_frequency(freq)

            # need to turn on MW itself again (optional)
            if self.mw_on_off:
                self.mw.start()

            # 1. signal data acquisition
            self._get_data()

            # 2. reference data acquisition (optional)
            if self.with_ref:
                # turn off MW via ASG (usually necessary)
                if self.asg_control_mw_on_off:
                    self.mw_control_seq([0, 0])

                # turn off MW itself (optional)
                if self.mw_on_off:
                    self.mw.stop()

                self._get_data_ref()

                # recover the sequences (usually necessary)
                if self.asg_control_mw_on_off:
                    self.mw_control_seq(mw_on_seq)

        print('finished data acquisition')

    def _acquire_data(self, *args, **kwargs):
        """
        Scanning time intervals to acquire data for Time-domain Scheduler
        :param with_ref: if True, MW 'on' and 'off' in turn
        """
        # 1. scan time intervals
        self._scan_freqs_and_get_data()

        # 2. calculate result (count with/without reference)
        self._cal_counts_result()

        # 3. save result
        self.save_result()

    def run_scanning(self, mw_control: str = 'on'):
        """
        Run the scheduler under scanning-frequency mode
        1) start device
        2) acquire data timely
        :param mw_control: 'on' or 'off'
        """
        self.is_running = True
        mw_seq_on = self.mw_control_seq()
        if mw_control == 'off':
            self.mw_control_seq([0, 0])
        elif mw_control == 'on':
            pass
        else:
            raise ValueError('unsupported MW control parameter (should be "on" or "off"')

        print('Begin to run {}. Frequency: {:.3f} - {:.3f} GHz.'.format(self.name, self._freqs[0] / C.giga,
                                                                        self._freqs[-1] / C.giga))
        print('t: {:.2f} ns, N: {}, T: {:.2f} s, n_freqs: {}'.format(self._asg_conf['t'] / C.nano, self._asg_conf['N'],
                                                                     self.asg_dwell, len(self._freqs)))
        print('Estimated total running time: {:.2f} s'.format(self.time_total))

        self._start_device()
        print('started')
        self._acquire_data()  # scanning MW frequencies in this loop
        self.stop()

        # recover the asg control sequence for MW to be 'on'
        if mw_control == 'off':
            self.mw_control_seq(mw_seq_on)
        self.is_running = False

    @abc.abstractmethod
    def configure_odmr_seq(self, *args, **kwargs):
        raise NotImplementedError

    def run_single_step(self, *args, **kwargs):
        raise NotImplementedError


class TimeDomainScheduler(Scheduler):
    """
    Time-domain ODMR detection abstract class
    """

    def __init__(self, *args, **kwargs):
        super(TimeDomainScheduler, self).__init__(*args, **kwargs)
        self.name = 'Time-domain ODMR Scheduler'

    def set_delay_times(self, start=None, end=None, step=None, times=None, length=None, logarithm=False):
        """
        Set time intervals for scanning detection (e.g. Ramsey, Rabi, DD, Relaxation)
        All unit is "ns"
        :param start: start time interval
        :param end: end time interval
        :param step: time interval step
        :param times: time duration list
        :param length: approximate number of time intervals; if designated, parameter `step` usually is not designated
        :param logarithm: whether to use exponential steps when `length` is designated; it is set True when the difference of `start` and `end` is large, e.g., T1 detection
        """
        if times is not None:
            self._times = list(times)
        elif step is not None:
            self._times = np.arange(start, end + step / 2, step).tolist()
        elif length is not None:
            if logarithm:
                self._times = np.unique(
                    (np.logspace(np.log10(start), np.log10(end), length) / 10).round() * 10).tolist()
            else:
                self._times = np.unique((np.linspace(start, end, length) / 10).round() * 10).tolist()
        else:
            raise ValueError('Please input sufficient parameters for time intervals generation')

        N = self._asg_conf['N']
        if N is None:
            raise ValueError('"N" is None currently. Please set ODMR sequences parameters firstly.')
        else:
            self.time_total = sum(self._times) * C.nano * N  # estimated total time
            if self.with_ref:
                self.time_total *= 2
            if self.order > 0:
                self.time_total *= self.order

    def _scan_times_and_get_data(self):
        """
        Scanning time intervals & getting data of Counter
        """
        # omit several scanning points
        for _ in range(self.epoch_omit):
            self.gene_detect_seq(self._times[0])
            self.asg.start()
            time.sleep(self.time_pad + self.asg_dwell)

            if self.with_ref:
                time.sleep(self.time_pad + self.asg_dwell)

        # formal data acquisition
        for duration in tqdm(self._times):
            self._cur_time = duration
            self.gene_detect_seq(duration)
            self.asg.start()

            # need to turn on MW itself again (optional)
            if self.mw_on_off:
                self.mw.start()

            # 1. signal data acquisition
            self._get_data()

            # 2. reference data acquisition
            if self.with_ref:
                # turn off MW via ASG (usually necessary)
                if self.asg_control_mw_on_off:
                    self.mw_control_seq([0, 0])

                # turn off MW itself (optional)
                if self.mw_on_off:
                    self.mw.stop()

                self._get_data_ref()

        print('finished data acquisition')

    def _acquire_data(self, *args, **kwargs):
        """
        Scanning time intervals to acquire data for Time-domain Scheduler
        """
        # 1. scan time intervals
        self._scan_times_and_get_data()

        # 2. calculate result (count with/without reference)
        self._cal_counts_result()

        # 3. save result
        self.save_result()

    def run_scanning(self):
        """
        Run the scheduler under scanning-time-interval mode
        1) start device
        2) acquire data timely
        """
        self.is_running = True
        print('Begin to run {}. Time intervals: {:.3f} - {:.3f} ns.'.format(self.name, self._times[0], self._times[-1]))
        print('N: {}, n_times: {}'.format(self._asg_conf['N'], len(self._times)))
        print('Estimated total running time: {:.2f} s'.format(self.time_total))

        self._start_device()
        self._acquire_data()  # scanning time intervals in this loop
        self.stop()
        self.is_running = False

    def gene_pseudo_detect_seq(self):
        """
        Generate pseudo pulses for visualization and regulation
        """
        ts = list(self._cache.values())[:-1]
        t_sum = sum([t for t in ts if t is not None])
        self.gene_detect_seq(int(t_sum / 4))

    @abc.abstractmethod
    def gene_detect_seq(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def configure_odmr_seq(self, *args, **kwargs):
        raise NotImplementedError

    def run_single_step(self, t_free) -> List[float]:
        """
        Single-interval & single-power setting for running the scheduler once
        :param t_free: free precession time, unit: ns
        :return: 1-D array-like data: [N,]
        """
        print('running with time = {:.3f} ns, MW power = {:.2f} dBm ...'.format(self.asg_dwell, self._mw_conf['power']))

        # generate ASG sequences
        self.gene_detect_seq(t_free)

        # start sequence for time: N * t
        self._start_device()
        time.sleep(2)  # let Laser and MW firstly start for several seconds
        time.sleep(self.asg_dwell)
        counts = self.counter.getData().ravel().tolist()
        self.stop()

        return counts
