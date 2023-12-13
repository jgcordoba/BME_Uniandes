
# f_SignalProcFuncLibs.py

from scipy import signal
import numpy as np

def AverageMean(dataArray,wind):
    average_y = []
    for ind in range(len(dataArray)):
        minPos = ind - wind
        maxPos = ind + wind
        if minPos < 0:
            minPos = 0
        if maxPos > len(dataArray) - 1:
            maxPos = len(dataArray) - 1
        average_y.append(np.mean(dataArray[minPos:maxPos]))
    return (average_y)


def f_GetIIRFilter(p_FsHz, p_PassFreqHz, p_StopFreqsHz):
    s_AMaxPassDb = 0.5
    s_AMinstopDb = 120
    s_NyFreqHz = p_FsHz / 2
    p_PassFreqHz = np.array(p_PassFreqHz) / s_NyFreqHz
    p_StopFreqsHz = np.array(p_StopFreqsHz) / s_NyFreqHz

    s_N, v_Wn = signal.cheb2ord(p_PassFreqHz, p_StopFreqsHz, s_AMaxPassDb, s_AMinstopDb)
    #print('f_GetIIRFilter - Filter order: ' + np.str(s_N))
    filt_FiltSOS = signal.cheby2(s_N, s_AMinstopDb, v_Wn, btype='bandpass', output='sos')

    return filt_FiltSOS

def f_IIRBiFilter(p_FiltSOS, p_XIn):
    return signal.sosfiltfilt(p_FiltSOS, p_XIn)

def f_FFTFilter(p_XIn, p_FsHz, p_FreqPassHz):
    s_EvenLen = 0
    s_N = np.size(p_XIn)

    if np.mod(s_N, 2.0) == 0:
        p_XIn = p_XIn[0:- 1]
        s_EvenLen = 1

    s_N = np.size(p_XIn)
    s_NHalf = np.int((s_N - 1) / 2)
    v_Freq = np.arange(0, s_N) * p_FsHz / s_N
    v_Freq = v_Freq[0:s_NHalf + 1]

    p_FreqPassHz = np.array(p_FreqPassHz)

    v_InputSigFFT = np.fft.fft(p_XIn)
    v_InputSigFFT = v_InputSigFFT[0:s_NHalf + 1]

    v_Ind = np.zeros(s_NHalf + 1)
    v_Ind = v_Ind > 0.0
    for s_Count in range(np.size(p_FreqPassHz, 0)):
        v_Ind1 = v_Freq >= p_FreqPassHz[s_Count, 0]
        v_Ind2 = v_Freq <= p_FreqPassHz[s_Count, 1]
        v_Ind = v_Ind + (v_Ind1 & v_Ind2)

    v_InputSigFFT[~v_Ind] = (10.0 ** -10.0) * np.exp(1j * np.angle(v_InputSigFFT[~v_Ind]))
    v_InputSigFFT = np.concatenate((v_InputSigFFT,np.flip(np.conjugate(v_InputSigFFT[1:]))))
    v_FiltSig = np.real(np.fft.ifft(v_InputSigFFT))

    if s_EvenLen:
        v_FiltSig = np.concatenate((v_FiltSig, [v_FiltSig[-1]]))

    return v_FiltSig


def f_GaborTFTransform(p_XIn, p_FsHz, p_F1Hz, p_F2Hz, p_FreqResHz, p_NumCycles):
    # Creamos un vector de tiempo en segundos
    v_TimeArray = np.arange(0, np.size(p_XIn)) / p_FsHz

    # Definimos un rango de frecuencias
    # las cuales usaremos para crear nuestros
    # patrones oscilatorios de prueba
    # En este caso generaremos patrones para
    # frecuencias entre 1 y 20 Hz con pasos
    # de 0.25 Hz
    v_FreqTestHz = np.arange(p_F1Hz, p_F2Hz + p_FreqResHz, p_FreqResHz)

    # Creamos una matriz que usaremos para
    # almacenar el resultado de las
    # convoluciones sucesivas. En esta matriz,
    # cada fila corresponde al resultado de
    # una convolución y cada columna a todos
    # los desplazamientos de tiempo.
    m_ConvMat = np.zeros([np.size(v_FreqTestHz), np.size(p_XIn)], dtype=complex)

    # Se obtiene la transformada de Fourier
    # de la señal p_XIn para usarla en cada iteración
    p_XInfft = np.fft.fft(p_XIn)

    # Ahora creamos un procedimiento iterativo
    # que recorra todas las frecuencias de prueba
    # definidas en el arreglo v_FreqTestHz
    for s_FreqIter in range(np.size(v_FreqTestHz)):
        # Generamos una señal sinusoidal de prueba
        # que oscile a la frecuencia de la iteración
        # s_FreqIter (v_FreqTestHz[s_FreqIter]) y que tenga
        # la misma longitud que la señal p_XIn.
        # En este caso usamos una exponencial compleja.
        xtest = np.exp(1j * 2.0 * np.pi * v_FreqTestHz[s_FreqIter] * v_TimeArray)

        # Creamos una ventana gaussina para
        # limitar nuestro patrón en el tiempo
        # Definimos la desviación estándar de
        # acuerdo al número de ciclos definidos
        # Dividimos entre 2 porque para un ventana
        # gaussiana, una desviación estándar
        # corresponde a la mitad del ancho de la ventana
        xtestwinstd = ((1.0 / v_FreqTestHz[s_FreqIter]) * p_NumCycles) / 2.0
        # Definimos nuestra ventana gaussiana
        xtestwin = np.exp(-0.5 * (v_TimeArray / xtestwinstd) ** 2.0)
        # Multiplicamos la señal patrón por
        # la ventana gaussiana
        xtest = xtest * xtestwin

        # Para cada sinusoidal de prueba obtenemos
        # el resultado de la convolución con la señal p_XIn
        # En este caso nos toca calcular la convolución
        # separadamente para la parte real e imaginaria
        # m_ConvMat[s_FreqIter, :] = np.convolve(p_XIn, np.real(xtest), 'same') + \
        #                        1j * np.convolve(p_XIn, np.imag(xtest), 'same')

        # Se obtine la transformada de Fourier del patrón
        fftxtest = np.fft.fft(xtest)
        # Se toma únicamente la parte real para evitar
        # corrimientos de fase
        fftxtest = abs(fftxtest)
        # Se obtine el resultado de la convolución realizando
        # la multiplicación de las transformadas de Fourier de
        # la señal p_XIn por la del patrón
        m_ConvMat[s_FreqIter, :] = np.fft.ifft(p_XInfft * fftxtest)

    return m_ConvMat, v_TimeArray, v_FreqTestHz



def f_bandpower(data, sf, band, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    l = len(data)

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nfft=l)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    idx_band_relative = np.logical_and(freqs >= 1, freqs <= 30)
    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd[idx_band_relative], dx=freq_res)
    return bp



def hilphase(y1,y2):
    sig1_hill=signal.hilbert(y1)
    sig2_hill=signal.hilbert(y2)
    pdt=(np.inner(sig1_hill,np.conj(sig2_hill))/(np.sqrt(np.inner(sig1_hill,
               np.conj(sig1_hill))*np.inner(sig2_hill,np.conj(sig2_hill)))))
    phase = np.angle(pdt)

    return phase