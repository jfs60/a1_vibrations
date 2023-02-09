import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy 
from scipy.fft import fft, fftfreq, ifft
from scipy.signal import get_window


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
    x = np.linspace(0,4,10001)
    return initial_signal_reshaped
    """plt.plot(x, initial_signal_reshaped, color="blue")
    plt.show()"""

def choose_range(time, initial_signal): 
    windowing_function = np.bartlett((time*(10001/30)))
    no_of_samples = int((10001/30)*time)
    x = np.linspace(0, time, no_of_samples)
    cut_signal = (initial_signal[0:no_of_samples])
    plt.plot(x, cut_signal, color="blue")
    plt.show()
    return cut_signal, no_of_samples, windowing_function




def applying_fft(signal_analysed, samples): 
    sample_rate = 10001/30
    duration = 4
    # Number of samples in normalized_tone
    N = int (sample_rate * duration)

    yf = fft(signal_analysed)
    xf = fftfreq(N, 1 / sample_rate)

    plt.plot(xf, np.abs(yf))
    plt.xlim([0, 5])   
    plt.ylim([0,0.1])
    plt.show()
    #not sure why they have different amplitude values 

def applying_window_function(window_function, signal_analysed, samples, time): 
    y = window_function*signal_analysed
    print(y)
    x = np.linspace(0, time, samples)
    plt.plot(x, y, color="blue")
    plt.show()



def main (): 
    initial_signal = setting_up()
    signal_analysed, samples, window_function = choose_range(4, initial_signal)
    print(signal_analysed.shape)
    applying_fft(signal_analysed, samples)
    applying_window_function (window_function, signal_analysed, samples, 4)
main()

 

