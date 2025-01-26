import wave
import numpy as np

#################################################################################################
# Before running change file_dir and file_name according to yours                               #
# Choose decimation factor (the bigger the factor the smaller the file and the resolution)      #
# Choose which interpolation (zoh, foh and zero padding) to run by switching to true or false   #
#################################################################################################

file_dir = '/Users/ohadperl/Desktop/BIU_work'
file_name = 'sample-15s.wav'

decimation_factor = 20  # Change this to the desired decimation factor

use_zero_padded = True
use_zoh = True
use_foh = True


def read_wav(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(params.nframes)
        audio_data = np.frombuffer(frames, dtype=np.int16)
    return audio_data, params


def write_wav(file_path, audio_data, params):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setparams(params)
        wav_file.writeframes(audio_data.tobytes())


def apply_lowpass_filter(audio_data, cutoff, fs, numtaps=101):
    if numtaps % 2 == 0:
        numtaps += 1

    nyquist = fs / 2
    norm_cutoff = cutoff / nyquist
    t = np.arange(-numtaps // 2 + 1, numtaps // 2 + 1)
    h = np.sinc(2 * norm_cutoff * t) * np.hamming(numtaps)
    h /= np.sum(h)

    filtered_data = np.convolve(audio_data, h, mode='same')
    return np.clip(filtered_data, -32768, 32767)


def decimate(audio_data, factor):
    return audio_data[::factor]


def zero_pedded_interp(x: np.ndarray, s: np.ndarray, u: np.ndarray):
    num_output = len(u)

    # Compute the FFT of the input signal
    X = np.fft.rfft(x)

    # Create a new array for the zero-padded frequency spectrum
    X_padded = np.zeros(num_output // 2 + 1, dtype=complex)

    # Copy the original frequency spectrum into the zero-padded array
    X_padded[:X.shape[0]] = X

    # Compute the inverse FFT of the zero-padded frequency spectrum
    x_interpolated = np.fft.irfft(X_padded, n=num_output)

    return x_interpolated * (num_output / len(s))


def zoh_interp(x, s, t):
    indices = np.searchsorted(s, t, side='right') - 1
    indices = np.clip(indices, 0, len(x) - 1)
    return x[indices]


def foh_interp(x, s, t):
    indices = np.searchsorted(s, t, side='right') - 1
    indices = np.clip(indices, 0, len(x) - 2)
    #It is calculated as the ratio of the distance from t to the previous point in s
    #over the distance between the two closest points in s
    frac = (t - s[indices]) / (s[indices + 1] - s[indices])
    #The interpolated values are a weighted average of the two closest points in x, with weights determined by frac
    return x[indices] * (1 - frac) + x[indices + 1] * frac

######################
# Main code execution#
######################


input_file = f'{file_dir}/{file_name}'
signal_max_freq = 20000  # Define the maximum frequency of the signal(highest frequency for human to hear)

audio_data, params = read_wav(input_file)

# new sample rate after decimation
new_sample_rate = params.framerate // decimation_factor #from all the samples we will take only the ones before the procces

# check if shanon is true
if new_sample_rate >= 2 * signal_max_freq: #nyquist
    filtered_data = audio_data
else:
    # הגדרת התדר החתוך למסנן אנטי-קיפול
    cutoff_freq = params.framerate / (2 * decimation_factor)
    # applynig anti-decimation filter
    filtered_data = apply_lowpass_filter(audio_data, cutoff_freq, params.framerate)

# decimation
decimated_data = decimate(filtered_data, decimation_factor)

# Adjust the sample rate parameter
new_params = list(params)
new_params[2] = new_sample_rate
new_params = tuple(new_params)

# create a time-line for the original
original_time = np.arange(len(audio_data)) / params.framerate

# create a time-line for the new one
decimated_time = np.arange(len(decimated_data)) * decimation_factor / params.framerate

#ZOH and  FOH
if use_zero_padded:
    sinc_reconstructed = zero_pedded_interp(decimated_data, decimated_time, original_time)
    sinc_output_file = f'{file_dir}/{file_name.replace(".wav", "_zero_padded.wav")}'
    write_wav(sinc_output_file, sinc_reconstructed.astype(np.int16), params)

if use_zoh:
    zoh_reconstructed = zoh_interp(decimated_data, decimated_time, original_time)
    zoh_output_file = f'{file_dir}/{file_name.replace(".wav", "_zoh.wav")}'
    write_wav(zoh_output_file, zoh_reconstructed.astype(np.int16), params)

if use_foh:
    foh_reconstructed = foh_interp(decimated_data, decimated_time, original_time)
    foh_output_file = f'{file_dir}/{file_name.replace(".wav", "_foh.wav")}'
    write_wav(foh_output_file, foh_reconstructed.astype(np.int16), params)

# Save the decimated data
decimated_output_file = f'{file_dir}/{file_name.replace(".wav", "_decimated.wav")}'
write_wav(decimated_output_file, decimated_data.astype(np.int16), new_params)
