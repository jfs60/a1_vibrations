import scipy 
import matplotlib.pyplot as plt
import numpy as np

from scipy.fft import fft, fftfreq

# Number of samples in normalized_tone

yf = fft(normalized_tone)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()