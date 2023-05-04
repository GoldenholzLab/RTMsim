# THIS IS A FAKE VERSION OF THE REALSIM.py 
# use this for testing only. NOT REAL!

import numpy as np
from realSim import get_mSF

def simulator_base_fake(sampRATE,number_of_days,cyclesTF=True,clustersTF=True, maxlimits=True, defaultSeizureFreq = -1,
    Lparams=[1.5475,17.645,10.617,5.9917,-0.3085,1.5371],CP=[],returnDetails=False,clusterParams=[.5, 1, 7, 1, 1],
    bestM=1.2267388224600906,bestS = 1.1457004817186776):

    SFparams = [defaultSeizureFreq, bestM, bestS]
    if defaultSeizureFreq==-1:
        mSF = get_mSF( requested_msf=SFparams[0],bestM=SFparams[1],bestS=SFparams[2] )
        #mSF = get_mSF(requested_msf=-1)
        
    SF = mSF / 30
    howmany = int(number_of_days*sampRATE)
    x = np.random.poisson(lam = SF, size=howmany)
    return x

def get_mSF(requested_msf,bestM=1,bestS=1):
    if requested_msf==-1:
        mSF = np.random.random()*9+1
    else:
        mSF = requested_msf
        
    return mSF


def downsample(x,byHowmuch):
    # input: 
    #    x = diary
    #    byHowMuch = integeter by how much to downsample
    # outputs
    #   x3 = the new diary, downsampled.
    #
    # If I sample 24 samples per day, and downsample by 24 then I get
    # daily samples as the output, for instance.
    #
    L = len(x)
    x2 = np.reshape(x,(int(L/byHowmuch),byHowmuch))
    x3 = np.sum(x2,axis=1)
    return x3