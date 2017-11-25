#Chinmay Kulkarni 

#Redid-820900828

import numpy as np
from mydsp.utils import get_corpus
import os
from mydsp.pca import PCA
from mydsp.plots import concatenated_spectrogram,pca_variance_captured,pca_gram
from mydsp.multifileaudioframes import MultiFileAudioFrames
from mydsp.dftstream import DFTStream

current_wd = os.getcwd() #get current working directory
files = get_corpus(current_wd) #get_corpus() returns list of files

files_22 = files[0:22] #taking first 22 files 
file_1 = [] #taking all wide bands
file_2 = [] #taking all narrow bands

for count in range(len(files_22)):
    if count%2==0 :
        file_1.append(files[count]) #append all a files
    else :
        file_2.append(files[count]) #append all b files

concatenated_spectrogram(file_1,10,20) #narrowband
concatenated_spectrogram(file_2,5,10) #wideband

files = sorted(files)
frame_s = MultiFileAudioFrames(files,10,20)
dft_stream = DFTStream(frame_s)
dft_list = [ d for d in dft_stream] #creating dft list
pca_dfts = PCA(dft_list)
get_val = pca_variance_captured(pca_dfts)
get_val = sorted(get_val)

l = []
for i,v in enumerate(get_val):
    while v >= 0.1*(len(l)+1):
        if 0.1*(len(l)+1) >0.9:
            break
        l.append(i+1)
        
for i,j in enumerate(l):
    print('{} components required to capture {:.1f} variance'.format(l[i],0.1*(i+1)))
    
vall = pca_dfts.transform(dft_list, l[6]) #calling trasnform function
taxis = 10*len(dft_list)
taxis = [i/1000 for i in range(0,taxis,10)] #time axis setup
pca_gram(np.transpose(vall), taxis)
            
        
    





