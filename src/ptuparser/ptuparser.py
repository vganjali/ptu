import os
import numpy as np
import h5py as h5py
import mmap as mmap
from scipy.sparse import csr_matrix
from tqdm import tqdm

class ptu(object):
    def __init__(self, filename='', binsize=1e-3, chunksize=100000):
        self.filename = filename
        self.binsize = binsize
        self.chunksize = chunksize
        self.globres = 250e-12

    def processHT2(self,filename=None,binsize=None):
        if type(filename) == str:
            self.filename = filename
        if binsize != None:
            self.binsize = binsize
        T2WRAPAROUND_V2 = 33554432
        ofcorr = 0
        bin_size = int(self.binsize/self.globres)
        bin_edge = {'time':0, 'count':0}
        self.timeout = 5
        with h5py.File(os.path.splitext(self.filename)[0]+'.hdf5', 'a') as fout:
            if (f'{self.binsize:010.6f}' not in fout.keys()):
                grp = fout.create_group(f'{self.binsize:010.6f}')
                ds_time = grp.create_dataset('time', (0,), chunks=(10000000,), maxshape=(None,), dtype=np.uint64, compression='lzf')
                ds_count = grp.create_dataset('count', (0,), chunks=(10000000,), maxshape=(None,), dtype=np.uint64, compression='lzf')
            else:
                ds_time = fout[f'{self.binsize:010.6f}']['time']
                ds_count = fout[f'{self.binsize:010.6f}']['count']
            with open(self.filename, mode='rb') as fin:
                f_mmap = mmap.mmap(fin.fileno(),0,access=mmap.ACCESS_READ)
                offset = int(f_mmap.find(str.encode('Header_End')))
                if offset == -1:
                    offset = 0
                    print('Warning: No header was found. Reading raw timetags...')
                else:
                    offset += 48
                total = int(os.path.getsize(self.filename)/(self.chunksize*4)+1)
                with tqdm(total=total) as pbar:
                    while (self.timeout > 0):
                        try:
                            records = np.frombuffer(f_mmap,dtype=np.uint32,count=self.chunksize,offset=offset)
                            special = np.bitwise_and(np.right_shift(records,31), 0x01).astype(np.byte)
                            channel = np.bitwise_and(np.right_shift(records,25), 0x3F).astype(np.byte)
                            timetag = np.bitwise_and(records, 0x1FFFFFF).astype(np.uint64)
                            overflowCorrection = np.zeros(self.chunksize, dtype=np.uint64)
                            overflowCorrection[np.where((channel == 0x3F) & (timetag == 0))[0]] = T2WRAPAROUND_V2
                            _loc = np.where((channel == 0x3F) & (timetag != 0))[0]
                            overflowCorrection[_loc] = np.multiply(timetag[_loc], T2WRAPAROUND_V2)
                            overflowCorrection = np.cumsum(overflowCorrection)+ofcorr
                            ofcorr = overflowCorrection[-1]
                            timetag += overflowCorrection
                            _tmp = timetag[np.where((special != 1) | (channel == 0x00))[0]].astype(np.uint64)
                            binned_sparse_time, binned_sparse_count = np.unique(np.insert(\
                                np.floor_divide(_tmp,bin_size),0,[bin_edge['time']]*bin_edge['count'])\
                                .astype(np.uint64),return_counts=True)
                            bin_edge['time'], bin_edge['count'] = binned_sparse_time[-1], binned_sparse_count[-1]
                            ds_count.resize((binned_sparse_time[-1]+1,))
                            dense_data = csr_matrix((binned_sparse_count,([0]*len(binned_sparse_time),binned_sparse_time-binned_sparse_time[0])), shape=(1,int(binned_sparse_time[-1]-binned_sparse_time[0]+1)), dtype=np.int32).toarray()[0]
                            ds_count[binned_sparse_time[0]:int(binned_sparse_time[-1]+1)] = dense_data
                            offset += self.chunksize*4
                            pbar.update(100/total)
                        except Exception as e:
                            if (len(f_mmap[offset:])/4 < self.chunksize):
                                self.chunksize = int(len(f_mmap[offset:])/4)
                            self.timeout -= 1
            ds_count[bin_edge['time']] = bin_edge['count']
            ds_time.resize((ds_count.size,))
            ds_time[:] = np.arange(0, self.binsize*1e12*ds_time.size, self.binsize*1e12, dtype=np.uint64)
        return