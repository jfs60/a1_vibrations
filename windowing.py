import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy 
from scipy.fft import fft, fftfreq, ifft


#setup the window function
def setting_up(): 
    initial_signal = []
    with (open("t_response.pickle", "rb")) as openfile:
        while True:
            try:
                initial_signal.append(pickle.load(openfile))
            except EOFError:
                break

    initial_signal_array = np.array(initial_signal)
    initial_signal_reshaped = initial_signal_array.ravel()
    x = np.linspace(0,30,10001)
    return initial_signal_reshaped
    """plt.plot(x, initial_signal_reshaped, color="blue")
    plt.show()"""

def choose_range(time, initial_signal): 
    windowing_function = np.bartlett(time+1)
    no_of_samples = int((10001/30)*time)
    x = np.linspace(0, time, no_of_samples)
    cut_signal = (initial_signal[0:no_of_samples])
    plt.plot(x, cut_signal, color="blue")
    plt.show()
    return cut_signal, no_of_samples

def freq_response(w_list, M, L, K, F):

    """Return complex frequency response of system"""

    return np.array(
        [np.linalg.solve(-w*w * M + 1j * w * L + K, F) for w in w_list]
    )

def last_nonzero(arr, axis, invalid_val=-1):

    """Return index of last non-zero element of an array"""

    mask = (arr != 0)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)






def applying_fft(signal_analysed, samples): 
    fft_signal = fft(signal_analysed)

    N = len(fft_signal)
    n = np.arange(N)
    T = N/(10)
    freq = n/T 

    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')

    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = freq[:n_oneside]

    # normalize the amplitude
    X_oneside =fft_signal[:n_oneside]/n_oneside

    plt.subplot(122)
    plt.stem(f_oneside, abs(X_oneside), 'b', \
            markerfmt=" ", basefmt="-b")
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Normalized FFT Amplitude |X(freq)|')
    plt.tight_layout()
    plt.show()
    #sample_spacing = 1/5


    """x = np.linspace(0.0, sample_spacing*samples, samples, endpoint=False)
    xf = fftfreq(samples, sample_spacing)[:samples]
    plt.plot(xf, 2.0/samples * np.abs(fft_signal[0:samples]))
    plt.grid()
    plt.show()"""




def main (): 
    initial_signal = setting_up()
    signal_analysed, samples = choose_range(30, initial_signal)
    print(signal_analysed.shape)
    applying_fft(signal_analysed, samples)
main()

 

