import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pyaudio
import librosa
import numpy as np
import pandas as pd
import warnings
from scipy import stats
from keras import Sequential
from keras.layers import Dense, Activation

def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:d}'.format(i + 1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(name, x, sr):
    features = pd.Series(index=columns(), dtype=np.float32, name=name)

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    f = librosa.feature.zero_crossing_rate(x, frame_length=2048, hop_length=512)
    feature_stats('zcr', f)

    cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12, tuning=None))
    assert cqt.shape[0] == 7 * 12
    assert np.ceil(len(x) / 512) <= cqt.shape[1] <= np.ceil(len(x) / 512) + 1

    f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cqt', f)
    f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
    feature_stats('chroma_cens', f)
    f = librosa.feature.tonnetz(chroma=f)
    feature_stats('tonnetz', f)

    del cqt
    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
    assert stft.shape[0] == 1 + 2048 // 2
    assert np.ceil(len(x) / 512) <= stft.shape[1] <= np.ceil(len(x) / 512) + 1
    del x

    f = librosa.feature.chroma_stft(S=stft ** 2, n_chroma=12)
    feature_stats('chroma_stft', f)

    f = librosa.feature.rmse(S=stft)
    feature_stats('rmse', f)

    f = librosa.feature.spectral_centroid(S=stft)
    feature_stats('spectral_centroid', f)
    f = librosa.feature.spectral_bandwidth(S=stft)
    feature_stats('spectral_bandwidth', f)
    f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
    feature_stats('spectral_contrast', f)
    f = librosa.feature.spectral_rolloff(S=stft)
    feature_stats('spectral_rolloff', f)

    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    del stft
    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    feature_stats('mfcc', f)

    features.index = features.index.map('_'.join)

    return np.expand_dims(features.transpose(), axis=0)


def get_model():
    model = Sequential()
    model.add(Dense(30, input_dim=518, init="uniform", activation="relu"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8))
    model.add(Activation("softmax"))
    model.load_weights("music_cat8_weight.h5")
    return model


def callback(in_data, frame_count, time_info, flag):
    audio_data = np.fromstring(in_data, dtype=np.float32)

    model = get_model()
    features = compute_features('test', audio_data, RATE)
    print(model.predict(features))

    return (in_data, pyaudio.paContinue)

RATE=44100
RECORD_SECONDS = 30
EVAL_SECONDS = 10
CHUNKSIZE = 2048
CATEGORIES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental','International', 'Pop', 'Rock']

pa = pyaudio.PyAudio()
print(pa.get_device_info_by_index(0)['defaultSampleRate'])
stream = pa.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNKSIZE)
model = get_model()

print("* Start Recording")
stream.start_stream()
frames = []

for i in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
    data = stream.read(CHUNKSIZE, exception_on_overflow=False)
    frames.append(np.fromstring(data, dtype=np.float32))

print("* End Recording")

audio_data = np.hstack(frames)
features = compute_features('test', audio_data, RATE)
pred = model.predict(features)
cat = np.argmax(pred)
print(pred)
print(CATEGORIES[cat])

# close stream
print("* End Recording")
stream.stop_stream()
stream.close()
pa.terminate()

# # start the stream
# stream.start_stream()
#
# while stream.is_active():
#     time.sleep(0.25)
#
# stream.close()
# pa.terminate()
