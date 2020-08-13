import sys
import matplotlib
import matplotlib.pyplot as plt2
import matplotlib.backends.backend_pdf as pdf
import astropy.units as u, astropy.constants as c
from astropy.time import Time
import Libfindgp_aro as gp
#import gp_polFit as pol
#import plot_gp_polFit as plot
from pylab import *
import os, sys, time, glob, itertools

from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#in_folder = '/scratch/drives/M/8/backup/B0531+21/20200112T183144Z_aro_vdif/20200205/'
#in_folder = '/scratch/p/pen/akanksha/recalled/20150724T080208Z_aro_raw/'
#in_folder = '/scratch/p/pen/akanksha/recalled/20180425T192145Z_aro_vdif/'
#in_folder = '/scratch/p/pen/akanksha/recalled/20190725T160852Z_aro_vdif/'
in_folder = '/scratch/p/pen/akanksha/recalled/drao_20150724T112127Z/'

#out_folder = '/scratch/akanksha/crabGP/20200205/'
#out_folder = '/scratch/p/pen/akanksha/crab_gp/gp_search/20150724T080208Z_1024/'
#out_folder = '/scratch/p/pen/akanksha/crab_gp/gp_search/20180425T192145Z_1024/'
#out_folder = '/scratch/p/pen/akanksha/crab_gp/gp_search/20190725T160852Z_3/'
out_folder = '/scratch/p/pen/akanksha/saved_triggers/drao_20150724T112127Z/'

#poly_file = '/scratch/akanksha/crabGP/20200205_polyco_new.dat'
#poly_file = '/scratch/p/pen/akanksha/crab_gp/polycos/B0531+21_20150724_polyco_new.dat'
poly_file = '/home/p/pen/akanksha/polycos/polyco_new_drao20150724.dat'
#poly_file = '/home/p/pen/akanksha/polycos/B0531+21_20180425_polyco_new.dat'
#poly_file = '/home/p/pen/akanksha/polycos/B0531+21_20190725_polyco_new.dat'

#DM = 56.754 #20180425
DM = 56.7703 #20150724
#files = '00001[0-1][0-9].dat'
#files = '0000000.dat'
#files = '000005[2-4].dat'
#files = sorted(glob.glob(in_folder + '*.vdif'))[40800:56000]
data_type = 1
#x = gp.AROPulsarAnalysis(in_folder, out_folder, data_type, poly_file, DM, fl=files)
x = gp.AROPulsarAnalysis(in_folder, out_folder, data_type, poly_file, DM)
print(len(x.files))
print(x.files[0])
print(x.files[len(x.files)-1])
print("****start time is ***", x.start_time)
print("****stop time is ****", x.stop_time)
#print("****start time is ***", x.start_time + 12*u.hr + 4*u.s)
#print("****stop time is ****", x.stop_time + 12*u.hr + 4*u.s)
#x.start_time = Time('2019-07-25T16:10:00') 
#x.stop_time = Time('2019-07-25T18:26:00')
#print('data_type is ', x.data_type)
#gp_timeStamp = Time('2015-07-24T18:42:01.711') - 0.26*u.s
#x.stop_time = Time('2015-07-24T17:05:00')
N = 2**20
#N = 2**19
ngate = 512
NFFT = 1

x.wrap += (-x.wrap) % NFFT
block_length = ((N - x.wrap) * x.dt).to(u.s)
max_time = ((x.stop_time - x.start_time) - x.wrap * x.dt).to(u.s)
max_blocks = int(floor((max_time / block_length).decompose().value))
print('max blocks: ', max_blocks)
num_blocks = max_blocks #260 
assert num_blocks <= max_blocks
timestamps = [x.start_time + i * block_length for i in range(num_blocks)]
#timestamps = [Time(Time((x.start_time + i * block_length), format='unix').isot) for i in range(num_blocks)]


if rank == 0:
    print(f"------------------------\n"
          f"Folding {x.psr_name} data.\n"
          f"Observation Details --\n"
          f"{x.start_time} -> {x.stop_time}\n"
          f"Total Duration (s): {max_time}\n"
          f"Block Length (s): {block_length.to(u.s)}\n"
          f"No. of blocks: {num_blocks} (Max: {max_blocks})\n"
#          f"Time to fold: {(num_blocks * block_length).to(u.s)}\n"
          f"------------------------", flush=True)

comm.Barrier()

time.sleep(rank)
block = 0
for timestamp in timestamps[rank::size]:
    print(f'{timestamp}')
#    print('block no: ', block)
#    pp, count = fold_band(x, timestamp, N, ngate, NFFT)
#    ppfull += pp
#    counts += count
    try:
    #x.read_data(N, timestamp)
        gp_data = gp.gp_finder_npz(x, timestamp, N, gp_thres=5, gp_size=512)
    except Exception as e:
        print(e)
        print('error in this timestamp {0}'.format(timestamp))
        pass
#    np.save('gp.npy', gp_data)
#    block = block + 1

print('Done the whole search!')

#z_raw = x.read_data(N, gp_timeStamp)
#print(z_raw.shape)
#z_raw = z_raw - z_raw.mean(0, keepdims=True)
#z_raw = z_raw - z_raw.std(0, keepdims=True)
#plot.plot_raw(z_raw, 128, gp_timeStamp, "raw.png", returnfig=0)
#!mkdir /scratch/p/pen/akanksha/crab_gp/gp_search/output_20150724T170055Z/saved
#z_raw = x.coh_dd(z_raw)
#print(z_raw.shape)
#plot.plot_raw(z_raw, 64, gp_timeStamp, "dd.png", returnfig=0)
