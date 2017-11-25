#Chinmay Kulkarni 

#Redid-820900828

from .multifileaudioframes import MultiFileAudioFrames
from .dftstream import DFTStream

import matplotlib.pyplot as plt
import numpy as np
print("entered plots")
def spectrogram(intensity, taxis, faxis):
    """spectrogram(intensity, taxis, faxis) - Plot an MxN intensity matrix
    where row i is frequency faxis(i) Hz and column j 
    is taxis(j) seconds
    
    If axis is specified, plot is created on the specified
    axis.
    """
    print(len(faxis))
    print(len(taxis))
    print(np.shape(intensity))
    
    
    fig, ax = plt.subplots()
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    k = ax.pcolormesh(taxis, faxis, intensity)
    c = plt.colorbar(k)
    c.set_label('Intensity (dB rel.)')
    plt.show()
    
def pca_gram(values, taxis):
    """pca_gram(values, taxis)
    Plot PCA transformed data in a manner similar to a spectrogram
    Each row is a component and columns form samples of PCA transformed
    data.  taxis is a vector of time units corresponding to each sample
    in seconds
    """
    figure,ax = plt.subplots()
    plt.xlabel('Time (S)')
    plt.ylabel('Number of components')
    faxis = [i for i in range(0,len(values))]
    k = ax.pcolormesh(taxis,faxis,values)
    plt.colorbar(k)
    plt.show()
    
    
def concatenated_spectrogram(files, adv_ms, len_ms):
    """concatenated_spectrogram(files, adv_ms, len_ms)
    Given a list of files, create a spectrogram of all files
    with frame advance of adv_ms and length of len_ms.
    
    Returns the spectra associated with the spectrogram
    """
    #print("entered ")
    #print("whatever")
    frames = MultiFileAudioFrames(files, adv_ms, len_ms)
    dfts = DFTStream(frames)
    faxis = dfts.get_Hz()
    dlist = []
    taxis = 0
    for i,d in enumerate(dfts):
        taxis += adv_ms
        dlist.append(d)
    
    taxis = [i/1000 for i in range(0,taxis,adv_ms)]	# map frame no to time
    
    faxis = faxis[:int(len(faxis)/2)]
    spectrogram(np.transpose(dlist), taxis, faxis)
    
def pca_variance_captured(pca):
    """"pca_variance_captured(pca)
    Given a PCA object, show the cumulative percentage ofvariance captured
    across principle components.
    
    Produces a plot with # of components on abscissa (x-axis)
    and % of variance captured on ordinate (y-axis, normalized [0,1])
    Returns the y values of the plot
    """
    
    #amount of variance captured m<=N components
    ax1 = (np.cumsum(pca.eigval) / np.sum(pca.eigval))
    plt.plot(ax1)
    plt.show()
    return ax1
       
    
    
    