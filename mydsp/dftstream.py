#Chinmay Kulkarni 

#Redid-820900828

import numpy as np
import scipy.signal as signal

print("dftstream")
class DFTStream:
    '''
    DFTStream - Transform a frame stream to various forms of spectra
    '''


    def __init__(self, frame_stream, specfmt="dB"):
        '''
        DFTStream(frame_stream, specfmt)        
        Create a stream of discrete Fourier transform (DFT) frames using the
        specified sample frame stream. Only bins up to the Nyquist rate are
        returned in the stream.
        
        Optional arguments:
        
        specfmt - DFT output:  
            "complex" - return complex DFT results
             "dB" [default] - return power spectrum 20log10(magnitude)
             "mag^2" - magnitude squared spectrum
        '''
        
        self.frame_stream = frame_stream #modified
        self.frame_it = iter(frame_stream)
        self.frame_stream_len = frame_stream.get_framelen_ms()/1000 #getting frame stream length
        self.Fs = frame_stream.get_Fs() #getting sampling rate
        self.num_frames = int(np.round(self.Fs * self.frame_stream_len))
        self.bins_Hz = np.arange(self.num_frames) / self.num_frames*self.Fs #bins in Hz
        self.bins_N = (i for i in self.bins_Hz)
        # Use self.bins_N to represent the number of bins returned
        
    def shape(self):
        "shape() - Return dimensions of tensor yielded by next()"
        return np.asarray([len(self.bins_N), 1])
    
    def size(self):
        "size() - number of elements in tensor generated by iterator"
        return np.asarray(np.product(self.shape()))
    
    def get_Hz(self):
        "get_Hz(Nyquist) - Return frequency bin labels"
        return self.bins_Hz
            
    def __iter__(self): #modified
        "__iter__() Return iterator for stream"
        return self
    
    def __next__(self): #modified
        "__next__() Return next DFT frame"
        framedata = next(self.frame_it)
        frame_length = len(framedata)
        window = signal.get_window("hamming", frame_length)
        windowed_x = framedata * window
        X = np.fft.fft(windowed_x)
        magX = np.abs(X)
        magX = magX[: int(len(magX)/2)]
        S = 20 * np.log10(magX)
        return (S)
        
        
    def __len__(self):
        "len() - Number of tensors in stream"
        return len(self.framer)

        
        
        
    
        