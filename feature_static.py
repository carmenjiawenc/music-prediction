import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from keras import Sequential
from keras.layers import Dense, Activation

DATA_DIR = '/Users/carmenc./Desktop/School/HKUST/MSBD5012'
DATA_META_DIR = os.path.join(DATA_DIR, 'fma_metadata')
DATA_AUDIO_DIR = os.path.join(DATA_DIR, 'fma_full')
CATEGORIES = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 'Instrumental','International', 'Pop', 'Rock']

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


def get_audio_path(audio_dir, track_id):
    """
    '../data/fma_small/000/000002.mp3'
    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def compute_features(tid):
    features = pd.Series(index=columns(), dtype=np.float32, name=tid)

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

    filepath = get_audio_path(DATA_AUDIO_DIR, tid)
    x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

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

    return features


def check_difference(tracks=None):
    train_features = pd.read_csv(os.path.join(DATA_DIR, 'MSBD 5012 Project', 'features_cat_8.csv'))
    train_features_sample = tracks
    count = 0
    for i in train_features_sample:
        gen_features = compute_features(i)
        df = pd.DataFrame(gen_features)
        df = df.join(train_features[train_features['track_id']==i][gen_features.index].transpose(), how='left')
        df.columns = ['gen', 'org']
        df['diff'] = abs(df['gen'] - df['org'])
        print(i)
        print(df)
        count += 1

def get_model():
    model = Sequential()
    model.add(Dense(30, input_dim=518, init="uniform", activation="relu"))
    model.add(Dense(40, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(8))
    model.add(Activation("softmax"))
    model.load_weights("music_cat8_weight.h5")
    return model

def main(track_id):
    # tracks = [2,5,10,140,141,148,182,190,193,200]
    # check_difference([2])
    # print("Track ID: %s"%track_id)

    model = get_model()
    for i in [0]:
        print("Track ID: %s" % i)
        features = compute_features(i)
        pred = model.predict(np.expand_dims(features.transpose(), axis=0))
        cat = np.argmax(pred)
        print(CATEGORIES[cat])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='song to generate features for, must be in tracks.csv')
    parser.add_argument('-t', '--trackid', action='store', dest='trackid', nargs=1)
    args = parser.parse_args()
    trackid = int(args.trackid[0])
    main(trackid)


