import scipy.io as sio
import numpy as np
from Functions.f_SignalProcFuncLibs import *
from visbrain.objects import TopoObj, ColorbarObj, SceneObj
from matplotlib.colors import ListedColormap

str_DataPath = 'Data/MatData/'  # path file where the data is stored
str_FileName = 'PSD.mat'  # Name of the File

dict_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
m_Data = dict_allData['m_Data']
d_SampleRate = np.double(dict_allData['d_SampleRate'][0])
v_ChanNames = np.array(dict_allData['v_ChanNames'])
v_FreqBands = np.array(dict_allData['v_FreqBands'])
v_SessionTimes = np.array(dict_allData['v_SessionTimes'][0])
d_Time1 = v_SessionTimes[0]
d_Time2 = v_SessionTimes[1]
v_FreqBands_Names = ['Delta', 'Alpha', 'Fast Beta']  # Names of the frequency bands
d_WindSec = 25
d_stepSec = 10
d_SampleRate = 1/2.5
d_WindSam = int(d_WindSec * d_SampleRate)
d_stepSam = int(d_stepSec * d_SampleRate)

m_AllCorrelation_Bef = []
m_AllCorrelation_Dur = []
m_AllCorrelation_Aft = []


for i_band in range(len(m_Data[0])):
    print(f'##################################################')
    print(f'PSD - processing freq band: {v_FreqBands_Names[i_band]}')
    print(f'##################################################')

    m_channCorrelation_Bef = np.zeros([len(m_Data), len(m_Data)])
    m_channCorrelation_Dur = np.zeros([len(m_Data), len(m_Data)])
    m_channCorrelation_Aft = np.zeros([len(m_Data), len(m_Data)])

    for i_chan1 in range(len(m_Data)):
        v_PSDEvolution1 = m_Data[i_chan1][i_band]

        for i_chan2 in range(len(m_Data)):
            v_PSDEvolution2 = m_Data[i_chan2][i_band]

            if i_chan1 != i_chan2:

                d_indexStart = 0
                d_indexEnd = int(d_indexStart + d_WindSam)
                v_PhaseCorr = []
                d_count = 0
                while d_indexEnd <= len(v_PSDEvolution1):
                    i_dataWind1 = v_PSDEvolution1[d_indexStart:d_indexEnd]
                    i_dataWind2 = v_PSDEvolution2[d_indexStart:d_indexEnd]

                    v_PhaseCorr.append(np.abs(hilphase(i_dataWind1, i_dataWind2)))
                    #corrpearson_Dr = pearsonr(v_FreqEvolution1_Dr, v_FreqEvolution2_Dr)[0]
                    d_count += 1

                    d_indexStart = int(d_indexStart + d_stepSam)
                    d_indexEnd = int(d_indexStart + d_WindSam)

                m_channCorrelation_Bef[i_chan1, i_chan2] = np.mean(v_PhaseCorr[0:int(v_SessionTimes[0] / d_stepSec)])
                m_channCorrelation_Dur[i_chan1, i_chan2] = np.mean(v_PhaseCorr[int(v_SessionTimes[0] / d_stepSec):int(v_SessionTimes[1] / d_stepSec)])
                m_channCorrelation_Aft[i_chan1, i_chan2] = np.mean(v_PhaseCorr[int(v_SessionTimes[1] / d_stepSec):])

            else:
                m_channCorrelation_Bef[i_chan1, i_chan2] = 0
                m_channCorrelation_Dur[i_chan1, i_chan2] = 0
                m_channCorrelation_Aft[i_chan1, i_chan2] = 0

    m_AllCorrelation_Bef.append(m_channCorrelation_Bef)
    m_AllCorrelation_Dur.append(m_channCorrelation_Dur)
    m_AllCorrelation_Aft.append(m_channCorrelation_Aft)



