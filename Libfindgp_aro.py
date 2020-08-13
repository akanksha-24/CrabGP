import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from pylab import *
import os, sys, time, glob, itertools
from astropy.time import Time
import astropy.units as u, astropy.constants as c
from baseband.helpers import sequentialfile as sf
from baseband import vdif
from pulsar.predictor import Polyco
from scipy.ndimage.filters import median_filter, uniform_filter1d
import pyfftw.interfaces.numpy_fft as fftw
from scintellometry.io import AROCHIMERawData

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
mstr = f'[{rank:2d}/{size:2d}]:'

_fftargs = {'threads': int(os.environ.get('OMP_NUM_THREADS', 2)), 
            'planner_effort': 'FFTW_ESTIMATE', 
            'overwrite_input': True}

D = 4148.808 * u.s * u.MHz**2 * u.cm**3 / u.pc

class AROPulsarAnalysis():
    """Class to analysis pulsar data at ARO.""" 
    def __init__(self, in_folder, out_folder, data_type, poly_file, DM, fl=None):
        '''The Crab'''
        self.psr_name = 'B0531+21'
        self.DM = DM * u.pc / u.cm**3
        self.folder = in_folder
        self.out_folder = out_folder
        self.data_type = data_type
        if fl != None:
           #self.filenames = fl
           self.files = fl
        else:
           if data_type == 0:
               self.filenames =  '*.dat'
           elif data_type == 1:
               self.filenames = '*.vdif'
           self.files = sorted(glob.glob(self.folder + self.filenames))[:]
        self.polyco = Polyco(poly_file)

        self.fref = 800. * u.MHz
        self.full_bw = 400. * u.MHz
        self.nchan = 1024
        self.npols = 2
        self.chan_bw = self.full_bw / self.nchan
        self.dt = (1 / self.chan_bw).to(u.s)
        self.sample_rate = 800.*u.MHz
        self.blocksize = 2**45
        self.fh = self.get_file_handle()
        if data_type == 0:
           self.start_time = Time(Time(self.fh.time0, format='unix').isot, precision=9)
           #fh_last = AROCHIMERawData([self.files[len(self.files)-1]], blocksize=self.blocksize, samplerate=self.sample_rate, 
           #                          fedge=self.fref, fedge_at_top=True)
           self.filesize = (self.fh.filesize * (len(self.files))) #- 1)) + fh_last.filesize
           #fh_last.close()
           self.stop_time = self.start_time + ((self.filesize / 2**11) * self.dt)
        elif data_type == 1:
           self.start_time = self.fh.start_time 
           self.stop_time = self.fh.stop_time
        self.fh.close()
        f0 = self.fref - self.full_bw
        wrap_time = (D * self.DM * (1/f0**2 - 1/self.fref**2)).to(u.s)
        wrap_samples = (wrap_time/self.dt).decompose().value
        self.wrap = int(np.ceil(wrap_samples))

        self.ftop = np.linspace(self.fref, self.fref - self.full_bw,
                                self.nchan, endpoint=False)

    def get_file_handle(self):
        """Returns file handle for a given list of channels."""
        if (self.data_type == 0):
           fh = AROCHIMERawData(self.files, blocksize=self.blocksize, samplerate=self.sample_rate,                                             fedge=self.fref, fedge_at_top=True) 
        elif (self.data_type == 1):
           fraw = sf.open(self.files, 'rb')
           fh = vdif.open(fraw, mode='rs', sample_rate=self.chan_bw, verify=False)
           #fh = vdif.open(self.files, mode='rs', sample_rate=self.chan_bw, verify=False)
        return fh

    def read_data(self, num_samples, timestamp):
        #fh = self.get_file_handle()
        fh = self.get_file_handle()
        if(self.data_type == 0):
           seek = (timestamp.unix - self.start_time.unix) * 8 * 10**8
           spectrum_bytes = 2**11
           try:
              start = int(seek//spectrum_bytes)*spectrum_bytes
              chunk = num_samples*spectrum_bytes  
              print("reading timestamp: ", timestamp)
              data = fh.seek_record_read(start, chunk)
              feed0 = data['f0'].astype(np.complex64)
              feed1 = data['f1'].astype(np.complex64)
              fh.close()
              return np.append(feed0[..., np.newaxis], feed1[..., np.newaxis], axis=2).transpose(0,2,1)
           except Exception as e:
              print(e)
              fh.close()
              #print("DEBUGGING HERE ** shape of data read: ", data.shape)
              #print("Shape of feed 1: ", data['f0'].shape)
              #print("Shape of feed 2: ", data['f1'].shape)
              #new_chunk = (fh.filesize * len(self.files) - start)
              #print('new end is: ', start + new_chunk)
              #data = fh.seek_record_read(start, new_chunk) 
        elif(self.data_type == 1):
           fh.seek(timestamp)
           z = fh.read(num_samples)
           fh.close()
           return z

    def find_outlier_threshold(self, x, n):
       """ Finds threshold for outliers. Function assumes that a clean signal will have mean 0."""

       assert 0 < x.ndim < 3 and n > 0
       x = x[..., np.newaxis] if x.ndim == 1 else x
       thres = []
       for x in x:
           s0 = np.std(x)
           if s0 != 0:
               s1 = np.std(x[abs(x) < n * s0])
               while not np.isclose(s0, s1):
                   s0 = s1
                   s1 = np.std(x[abs(x) < n * s0])
               thres += [n * s1]
           else:
               thres += [n]
       return thres 

    def old_find_outlier_threshold(self, x, n):
        """ Finds threshold for outliers. Function assumes that a clean signal will have mean 0."""

        assert 0 < x.ndim < 3
        x = x[..., np.newaxis] if x.ndim == 1 else x
        thres = []
        for x in x:
            s0 = np.std(x)
            s1 = np.std(x[abs(x) < n*s0])
            while not np.isclose(s0, s1):
                s0 = s1
                s1 = np.std(x[abs(x) < n*s0])
            thres += [n*s1]
        return thres  

    def remove_rfi(self, z, freq_smoothing=64, time_smoothing=1024, 
                    nstd=5, cutoff_factor=0):
        """ Remove RFI from a signal """
#        print('z.shape', z.shape)
        y = z.real**2 + z.imag**2
#        print('y.shape',y.shape)
        # All channels are good channels until they become bad channels!
        good_channels = np.ones(y.shape[-1], dtype=bool)
        # Finding mean power in channels
        mean_freq_power = y.mean(0)
        smooth_mean_freq_power = np.zeros_like(mean_freq_power)

        '''model'''
        for i in range(self.npols):
            smooth_mean_freq_power[i] = median_filter(mean_freq_power[i],
                                                      freq_smoothing, 
                                                      mode='mirror')
        # Normalizing mean power of channels and recomputing power in channels
        z = np.divide(z, smooth_mean_freq_power[np.newaxis], out=np.zeros_like(z), where=smooth_mean_freq_power[np.newaxis]!=0)
        y = z.real**2 + z.imag**2
        # Finding and tagging extra-bright channels as bad channels!
        mean_freq_power = y.mean(0)
        smooth_mean_freq_power = np.zeros_like(mean_freq_power)
        for i in range(self.npols):
            smooth_mean_freq_power[i] = median_filter(mean_freq_power[i],
                                                      freq_smoothing, 
                                                      mode='mirror')
        res = mean_freq_power - smooth_mean_freq_power
        bright_channels = abs(res).T > self.find_outlier_threshold(res, nstd)
        good_channels[bright_channels.any(-1)] = False
        var_freq_power = y.var(0)
        smooth_var_freq_power = np.zeros_like(var_freq_power)
        for i in range(self.npols):
            smooth_var_freq_power[i] = median_filter(var_freq_power[i],
                                                     freq_smoothing,
                                                     mode='mirror')
        res = var_freq_power - smooth_var_freq_power
        variable_channels = abs(res).T > self.find_outlier_threshold(res, nstd)
        good_channels[variable_channels.any(-1)] = False
        # Excising bad channels, and recomputing power
        z *= good_channels[np.newaxis, np.newaxis, ...]
        y = z.real**2 + z.imag**2
        # Finding time variability and normalizing it
        mean_time_power = y[..., good_channels].mean(-1)
        smooth_mean_time_power = uniform_filter1d(mean_time_power,
                                                  time_smoothing,
                                                  axis=0)
        # Normalizing power in time
        z = np.divide(z, smooth_mean_time_power[:,:,np.newaxis], out=np.zeros_like(z), where=smooth_mean_time_power[:,:,np.newaxis]!=0)
        y = z.real**2 + z.imag**2
        print('remove rfi done')
        return z


    def coherent_dedispersion(self, z, channel, axis=0):
        """Coherently dedisperse signal."""

        fcen = self.ftop[channel]
        tag = "{0:.2f}-{1:.2f}M_{2}".format(self.fref.value, fcen.value,
                                            z.shape[axis])
        ddcoh_file = self.out_folder + "saved/ddcoh_{0}.npy".format(tag)
        try:
            dd_coh = np.load(ddcoh_file)
        except:
            f = fcen + np.fft.fftfreq(z.shape[axis], self.dt)
            dang = D * self.DM * u.cycle * f * (1./self.fref - 1./f)**2
            with u.set_enabled_equivalencies(u.dimensionless_angles()):
                dd_coh = np.exp(dang * 1j).conj().astype(np.complex64).value
            np.save(ddcoh_file, dd_coh)
        if z.ndim > 1:
            ind = [np.newaxis] * z.ndim
           #ind[axis] = tuple(slice(None))
            ind[axis] = slice(None)
        if z.ndim > 1:
          try:
             dd_coh = dd_coh[ind]
          except:   
             print("z.shape", z.shape) 
             print("z.ndim, ", z.ndim)
             print("ind: ", ind)
             print("dd_coh.shape", dd_coh.shape)
        z = fftw.fft(z, axis=axis, **_fftargs)
        z = fftw.ifft(z * dd_coh, axis=axis, **_fftargs)
#        z = np.fft.fft(z, axis=axis)
#        z = np.fft.ifft(z * dd_coh, axis=axis)
        return z

    def coh_dd(self, z, wrap=True):
        for channel in range(self.nchan):
            if (self.data_type == 1):
                z[..., channel] = self.coherent_dedispersion(z[..., channel], channel)
            elif (self.data_type == 0):
                z[..., channel] = self.coherent_dedispersion(z[..., channel].conj(), channel)
        if wrap:
            z = z[:-self.wrap]
        return z

    def prep_forSearch(self, z_raw, int_param):
        z = abs(z_raw)**2
        z = z.sum(-1).sum(-1)
        #integration param
        #z = rebin(z, int_param, 1)
        #remove noise
        #z = z - z.mean(0, keepdims = True)
        #z = z / z.std(0, keepdims = True)
        return z 

    def process_file_test1(self, timestamp, num_samples):
        """Seeks, reads and dedisperses signal from a given timestamp"""
        fh = self.get_file_handle()
        print(f'print fh.shape: {fh.shape}')
        fh.seek(timestamp)
        print(f'print fh.seek: {fh.seek(timestamp)}')
#        print(timestamp)
        print(f'print num_samples: {num_samples}')
        z = fh.read(num_samples).astype(np.complex64)
        return z

    def process_file_subband(self, timestamp, num_samples):
    
        t0 = time.time()
        fh = self.get_file_handle()
        fh.seek(timestamp)


    def process_file(self, timestamp, num_samples):
        """Seeks, reads and dedisperses signal from a given timestamp"""

        if num_samples <= self.wrap:
            raise Exception(f'num_samples must be larger than {self.wrap}!')
        else:
            t0 = time.time()
            z = self.read_data(num_samples, timestamp)
            print ('z_original.shape', z.shape)
            t1 = time.time()
            print(f'{mstr} Took {t1 - t0:.2f}s to read.')
            
            t2 = time.time()
            z = self.coh_dd(z) 
            t3 = time.time()
            print(f'{mstr} Took {t3 - t2:.2f}s to dedisperse.')
        print ('z return shape', z.shape)
        return z

    def get_phases(self, timestamp, num_samples, dt, ngate):
        """Returns pulse phase."""

        phasepol = self.polyco.phasepol(timestamp, rphase='fraction', 
                                        t0=timestamp, time_unit=u.second,
                                        convert=True)
        ph = phasepol(np.arange(num_samples) * dt.to(u.s).value)
        ph -= np.floor(ph[0])
        ph = np.remainder(ph * ngate, ngate).astype(np.int32)
        return ph

    def gp_finder_method(self, z, gp_thres, gp_size):
        """Method to find giant pulses in signal."""
#        y = uniform_filter1d(z, gp_size, origin=-gp_size//2)
        y=z
        y = (y - y.mean()) / y.std()
        y /= y[abs(y) < 6].std()
        y /= y[abs(y) < 6].std()
        gp_index = np.argwhere(y > gp_thres).squeeze(-1)
        if gp_index.shape[0] > 0:
            gp_index = gp_index[np.logical_and(gp_index > gp_size,
                                               gp_index <
                                               (y.shape[0] - gp_size))]
            l0, l1 = 1, 0
            while l0 != l1:
                l0 = len(gp_index)
                for i, p in enumerate(gp_index):
                    gp_index[i] = (np.argmax(y[p-gp_size:p+gp_size]) + p
                                   - gp_size)
                gp_index = np.unique(gp_index)
                gp_index = gp_index[np.logical_and(gp_index > gp_size,
                                                   gp_index <
                                                   (y.shape[0] - gp_size))]
                l1 = len(gp_index)
        gp_sn = y[gp_index]
        return gp_index, gp_sn

    def get_times_list(self, num_samples):
        """Make time list."""

        first_file_num = True
        for file_num in self.config.file_nums:
            st = int(np.ceil(self.config.time_bounds[file_num][0].unix + 1))
            et = int((self.config.time_bounds[file_num][1] - (self.config.wrap
                 * self.config.dt)).unix - 1)
            file_length = (et - st) * u.s
            assert file_length > 0
            chuck_length = (num_samples - self.config.wrap) * self.config.dt
            num_chunks = int((file_length / chuck_length).decompose())
            times = np.linspace(st, et, num_chunks, endpoint=False)
            if first_file_num:
                tlist = np.array([Time(t, format='unix', precision=9) for t in times])
                for t in tlist:
                    t.format = 'isot'
                first_file_num = False
            else:
                tlist1 = np.array([Time(t, format='unix', precision=9) for t in times])
                for t in tlist1:
                    t.format = 'isot'
                tlist = np.concatenate((tlist, tlist1))
        return tlist

def gp_finder(pa, t, num_samples, gp_thres=5, gp_size=128):
    """Find giant pulses in signal."""

#    assert ((isinstance(tels, list) or isinstance(tels, str))
#            and len(tels) != 0)
#    if isinstance(tels, list) and len(tels) == 1:
#            tels = tels[0]
#    if isinstance(tels, list) and len(tels) != 1:
#        polyco = self.config.get_polyco(self.config.ref_tel)
#    else:
#        polyco = self.config.get_polyco(tels)

    print(f'{mstr} --Finding giant pulses--')
    gp_data = []
    gp_raw = []
    NFFT = 1
    phasepol = pa.polyco.phasepol(t, rphase='fraction', t0=t,
                               time_unit=u.second, convert=True)
#    ph = phasepol(np.arange(num_samples) * dt.to(u.s).value)
#    ph -= np.floor(ph[0])
#    ph = np.remainder(ph * ngate, ngate).astype(np.int32)


#    if isinstance(tels, list) and len(tels) != 1:
#        z = self.incoherently_stack_telescopes(t, tels, num_samples)
#    else:
#        z = self.process_signal.process_data(t, tels, num_samples,
#                                                 remove_rfi=True)[0]
    print('str(t)',str(t))
    z_raw = pa.process_file(t, num_samples) 
    z = abs(z_raw)**2
    z = z.sum(-1).sum(-1)
    print('z.shape',z.shape)
    print('gp_thres',gp_thres)
    print('gp_size',gp_size)
    gp_index, gp_sn = pa.gp_finder_method(z, gp_thres, gp_size)
    print((f'{mstr} -- Time: {t.isot} '
           f'Found {len(gp_index)} giant pulses.'))
    if len(gp_index) > 0:
        for index, sn in zip(gp_index, gp_sn):
            print('index: ',index)
            gp = phasepol((index * NFFT*pa.dt).to(u.s).value) #% 1
            event_t = t + index * pa.dt
            gp_data.append((event_t, sn, gp))
            if index >=2500:
                gp_raw.append((z_raw[(-2500+index):(index+2500)]))
            else:
                gp_raw.append((z_raw[0:(index+2500)]))
            trigger_info = [t.value, index, event_t.value, sn, gp]
            trigger_info = np.asarray(trigger_info)
            '''Save information of triggers'''
            triggers_file = pa.out_folder + 'triggers_aro_gp.npy'
            if os.path.exists(triggers_file):
                sequence = np.load(triggers_file)
                np.save(triggers_file, np.vstack((sequence, trigger_info)))
            else:
                np.save(triggers_file, trigger_info)
        file_name= pa.out_folder + 'gp_raw_t'+str(t)+'.npy'
        np.save(file_name, np.asarray(gp_raw))
        print('\x1b[6;30;43m' + '*** saved gp_raw ***' + '\x1b[0m')
    print(gp_data)
    return gp_data

def gp_finder_npz(pa, t, num_samples, gp_thres=5, gp_size=128):
    """Find giant pulses in signal."""

    print(f'{mstr} --Finding giant pulses--')
    gp_data = []
#    gp_raw = []
    NFFT = 1
    phasepol = pa.polyco.phasepol(t, rphase='fraction', t0=t,
                               time_unit=u.second, convert=True)
    print('str(t)',str(t))
    z_raw = pa.process_file(t, num_samples)
    z = abs(z_raw)**2
    z = z.sum(-1).sum(-1)
    print('z.shape',z.shape)
    print('gp_thres',gp_thres)
    print('gp_size',gp_size)
    gp_index, gp_sn = pa.gp_finder_method(z, gp_thres, gp_size)
    print((f'{mstr} -- Time: {t.isot} '
           f'Found {len(gp_index)} giant pulses.'))
    width = 5000
    if len(gp_index) > 0:
        for index, sn in zip(gp_index, gp_sn):
            print('index: ',index)
            gp = phasepol((index * NFFT*pa.dt).to(u.s).value) #% 1
            event_t = t + index * pa.dt
            gp_data.append((event_t, sn, gp))
            if index < width:
                 gp_raw = z_raw[0:width]
            elif index > (num_samples - width):
                 gp_raw = z_raw[num_samples-width:(num_samples-1)]
            else:
                 gp_raw = z_raw[int(index - width/2):int(index + width/2)]
            file_name = pa.out_folder + 'gp_raw_t'+str(event_t)+'.npz'
            np.savez(file_name, t=t, index=index, event_t=event_t, sn=sn, gp=gp, data=gp_raw)
            print('\x1b[6;30;43m' + '*** saved gp_raw ***' + '\x1b[0m')            

    print(gp_data)
    return gp_data

def find_triggers_npy(pa, t, z_raw, z, gp_thres=5, gp_size=128):
    print('--Finding giant pulses--')
    gp_data = []
    gp_raw = []
    NFFT = 1
    phasepol = pa.polyco.phasepol(t, rphase='fraction', t0=t,
                               time_unit=u.second, convert=True)
    gp_index, gp_sn = pa.gp_finder_method(z, gp_thres, gp_size)
    print((f'{mstr} -- Time: {t.isot} '
           f'Found {len(gp_index)} giant pulses.'))    
    if len(gp_index) > 0:
        for index, sn in zip(gp_index, gp_sn):
            print('index: ',index)
            gp = phasepol((index * NFFT*pa.dt).to(u.s).value) #% 1
            event_t = t + index * pa.dt
            gp_data.append((event_t, sn, gp))
            if index >=2500:
                gp_raw.append((z_raw[(-2500+index):(index+2500)]))
            else:
                gp_raw.append((z_raw[0:(index+2500)]))
            trigger_info = [t.value, index, event_t.value, sn, gp]
            trigger_info = np.asarray(trigger_info)
            '''Save information of triggers'''
            triggers_file = pa.out_folder + 'triggers_aro_gp.npy'
            if os.path.exists(triggers_file):
                sequence = np.load(triggers_file)
                np.save(triggers_file, np.vstack((sequence, trigger_info)))
            else:
                np.save(triggers_file, trigger_info)
        file_name= pa.out_folder + 'gp_raw_t'+str(t)+'.npy'
        np.save(file_name, np.asarray(gp_raw))
        print('\x1b[6;30;43m' + '*** saved gp_raw ***' + '\x1b[0m')
    print(gp_data)
    return gp_data 


def find_triggers_npz(pa, t, z_raw, z, gp_thres=5, gp_size=128):
    print('--Finding giant pulses--')
    gp_data = []
    gp_raw = []
    NFFT = 1
    phasepol = pa.polyco.phasepol(t, rphase='fraction', t0=t,
                               time_unit=u.second, convert=True)
    gp_index, gp_sn = pa.gp_finder_method(z, gp_thres, gp_size)
    print((f'{mstr} -- Time: {t.isot} '
           f'Found {len(gp_index)} giant pulses.'))
    width = 5000
    if len(gp_index) > 0:
        t_arr = []; indx_arr = []; event_arr = []; sn_arr = []; gp_arr =[]; data_arr =[];
        for index, sn in zip(gp_index, gp_sn):
            print('index: ',index)
            gp = phasepol((index * NFFT*pa.dt).to(u.s).value) #% 1
            event_t = t + index * pa.dt
            gp_data.append((event_t, sn, gp))
            if index < width:
                gp_raw.append((z_raw[0:width]))
            elif index > (num_samples - width):
                gp_raw.append((z_raw[num_samples-width:(num_samples-1)]))
            else:
                gp_raw.append((z_raw[int(index - width/2):int(index + width/2)]))
            t_arr.append(t.value)
            indx_arr.append(index)
            event_arr.append(event_t)
            sn_arr.append(sn)
            gp_arr.append(gp)
        t_arr = np.asarray(t_arr)
        indx_arr = np.asarray(indx_arr)
        event_arr = np.asarray(event_arr)
        sn_arr = np.asarray(sn_arr)
        gp_arr = np.asarray(gp_arr)
        data_arr = np.asarray(gp_raw)
        file_name = pa.out_folder + 'gp_raw_t'+str(t)+'.npz'
        np.savez(file_name, t=t_arr, index=indx_arr, event_t=event_arr, sn=sn_arr, gp=gp_arr, data=data_arr)
        print('\x1b[6;30;43m' + '*** saved gp_raw ***' + '\x1b[0m')
    print(gp_data)
    return gp_data

def make_waterfall(pa, timestamp, num_samples, tbin=1024):
    fh = pa.get_file_handle()
    fh.seek(timestamp)
    t0 = time.time()
    z = fh.read(num_samples).astype(np.complex64)
    t1 = time.time()
    print(f'{mstr} Took {t1 - t0:.2f}s to read data.')
    t2 = time.time()
    for channel in range(pa.nchan):
        z[..., channel] = pa.coherent_dedispersion(z[..., channel], channel)
    t3 = time.time()
    print(f'{mstr} Took {t3 - t2:.2f}s to dedisperse.')
    wrap = pa.wrap + (-pa.wrap % tbin)
    z = z[:-wrap]
    z = (z.real**2 + z.imag**2).astype(np.float32)
    z = z.reshape(-1, tbin, 2, 1024).mean(1)
    return z

def fold_band(pa, timestamp, num_samples, ngate, NFFT):
    z = pa.process_file(timestamp, num_samples)
    t0 = time.time()
    z = fftw.fft(z.reshape(-1, NFFT, pa.npols, pa.nchan), axis=1, **_fftargs)
    z = fftw.fftshift(z, axes=(1,))
#    z = np.fft.fft(z.reshape(-1, NFFT, pa.npols, pa.nchan), axis=1)
#    z = np.fft.fftshift(z, axes=(1,))
    z = (z.real**2 + z.imag**2).sum(2).astype(np.float32)
    z = z.transpose(0, 2, 1).reshape(z.shape[0], -1)
    ph = pa.get_phases(timestamp, z.shape[0], NFFT*pa.dt, ngate)
    count = np.bincount(ph, minlength=ngate)
    pp = np.zeros((ngate, z.shape[-1]))
    for channel in range(z.shape[-1]):
        pp[..., channel] = np.bincount(ph, z[..., channel], minlength=ngate)
    t1 = time.time()
    print(f'{mstr} Took {t1 - t0:.2f}s to fold 1 block.')

def rebin(matrix, xbin, ybin):
    data = np.zeros((math.ceil(matrix.shape[0]/xbin),
                     math.ceil(matrix.shape[1]/ybin)))
    for ii in range(data.shape[0]):
        for jj in range(data.shape[1]):
            data[ii,jj] = matrix[int(ii*xbin):int((ii+1)*xbin),int(jj*ybin):int((jj+1)*ybin)].mean()
    return data
