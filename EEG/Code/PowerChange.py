import scipy.io as sio
import numpy as np
from visbrain.objects import TopoObj, ColorbarObj, SceneObj


str_DataPath = 'Data/MatData/'  # path file where the data is stored
str_FileName = 'PSD.mat'  # Name of the File

dict_allData = sio.loadmat(str_DataPath + str_FileName)  # Load .mat data
m_Data = dict_allData['m_Data']
d_SampleRate = np.double(dict_allData['d_SampleRate'][0])
v_ChanNames = np.array(dict_allData['v_ChanNames'])
v_FreqBands = np.array(dict_allData['v_FreqBands'])
v_SessionTimes = np.array(dict_allData['v_SessionTimes'][0])
v_FreqBands_Names = ['Delta', 'Alpha', 'Fast Beta']  # Names of the frequency bands
s_windStep = 2.5
m_DiffData1 = []
m_DiffData2 = []
for i_chann in range(len(m_Data)):
    v_Data = m_Data[i_chann]
    v_MeanData_Bef = np.mean(v_Data[:, 0:int(v_SessionTimes[0] / s_windStep)], 1)
    v_MeanData_Dur = np.mean(v_Data[:, int(v_SessionTimes[0] / s_windStep):int(v_SessionTimes[1] / s_windStep)], 1)
    v_MeanData_aft = np.mean(v_Data[:, int(v_SessionTimes[1] / s_windStep):], 1)

    v_DiffData1 = v_MeanData_Dur - v_MeanData_Bef
    m_DiffData1.append(v_DiffData1)
    v_DiffData2 = v_MeanData_aft - v_MeanData_Bef
    m_DiffData2.append(v_DiffData2)

m_DiffData1 = np.array(m_DiffData1)
m_DiffData2 = np.array(m_DiffData2)


sc = SceneObj(bgcolor='white', size=(400 * (len(v_FreqBands_Names)-1), 350))
for i_band in range(len(v_FreqBands_Names)):
    v_Data = m_DiffData1[:, i_band]
    d_clim = np.max(np.abs([np.min(v_Data), np.max(v_Data)]))
    v_clim = [-d_clim, d_clim]
#'PuBu'
    kw_top = dict(margin=30 / 100, chan_offset=(0.1, 0.1, 0.), chan_size=8, levels=7, cmap='turbo',
                  level_colors='k',clim = v_clim )
    kw_cbar = dict(cbtxtsz=10, txtsz=10., width=.5, txtcolor='black', cbtxtsh=1.8, rect=(0., -1.5, 1., 3),
                   border=True)
    kw_title = dict(title_color='black', title_size=9.0, width_max=300)
    ch_names = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']

    t_obj = TopoObj('topo', v_Data, channels=ch_names, **kw_top)
    sc.add_to_subplot(t_obj, row=0, col=i_band, title=f' {v_FreqBands_Names[i_band]} - Tarea 1', **kw_title)


cb_obj1 = ColorbarObj(t_obj, cblabel='Z-Score PSD', **kw_cbar)
sc.add_to_subplot(cb_obj1, row=0, col=(i_band + 1), width_max=45)

for i_band in range(len(v_FreqBands_Names)):
    v_Data = m_DiffData2[:, i_band]
    d_clim = np.max(np.abs([np.min(v_Data), np.max(v_Data)]))
    v_clim = [-d_clim, d_clim]
#'PuBu'
    kw_top = dict(margin=30 / 100, chan_offset=(0.1, 0.1, 0.), chan_size=8, levels=7, cmap='turbo',
                  level_colors='k',clim = v_clim )
    kw_cbar = dict(cbtxtsz=10, txtsz=10., width=.5, txtcolor='black', cbtxtsh=1.8, rect=(0., -1.5, 1., 3),
                   border=True)
    kw_title = dict(title_color='black', title_size=9.0, width_max=300)
    ch_names = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']

    t_obj = TopoObj('topo', v_Data, channels=ch_names, **kw_top)
    sc.add_to_subplot(t_obj, row=1, col=i_band, title=f' {v_FreqBands_Names[i_band]} - Tarea 2', **kw_title)

cb_obj1 = ColorbarObj(t_obj, cblabel='Z-Score PSD', **kw_cbar)
sc.add_to_subplot(cb_obj1, row=1, col=(i_band + 1), width_max=45)


sc.preview()
