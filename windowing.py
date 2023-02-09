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




def applying_fft(signal_analysed, samples): 
    sample_rate = 10001/30
    duration = 30
    # Number of samples in normalized_tone
    N = int (sample_rate * duration)

    yf = fft(signal_analysed)
    xf = fftfreq(N, 1 / sample_rate)

    plt.plot(xf, np.abs(yf))
    plt.show()
""" N = len(fft_signal)
    n = np.arange(N)
    T = N/(samples/)
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
    #sample_spacing = 1/5"""


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

 

