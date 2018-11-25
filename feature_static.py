import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
import librosa
from util import compute_features, get_model, pca, CATEGORIES

DATA_DIR = '/Users/carmenc./Desktop/School/HKUST/MSBD5012'
DATA_META_DIR = os.path.join(DATA_DIR, 'fma_metadata')
DATA_AUDIO_DIR = os.path.join(DATA_DIR, 'fma_full')


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def check_difference(tracks=None):
    train_features = pd.read_csv('features_cat_6.csv')
    train_features_sample = tracks
    count = 0
    for i in train_features_sample:
        filepath = get_audio_path(DATA_AUDIO_DIR, i)
        x, sr = librosa.load(filepath, sr=None, mono=True)
        gen_features = compute_features(i, x, sr)
        df = pd.DataFrame(gen_features)
        df = df.join(train_features[train_features['track_id'] == i][gen_features.index].transpose(), how='left')
        df.columns = ['gen', 'org']
        df['diff'] = abs(df['gen'] - df['org'])
        print(i)
        print(df)
        count += 1


def main(track_id):
    # tracks = [2,5,10,140,141,148,182,190,193,200]
    # check_difference([2])
    # print("Track ID: %s"%track_id)

    model = get_model()
    print("Track ID: %s" % track_id)
    filepath = get_audio_path(DATA_AUDIO_DIR, track_id)
    x, sr = librosa.load(filepath, sr=None, mono=True)
    features = compute_features(track_id, x, sr)
    features = pca(features)
    pred = model.predict(features)
    cat = np.argmax(pred)
    print(CATEGORIES[cat])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='song to generate features for, must be in tracks.csv')
    parser.add_argument('-t', '--trackid', action='store', dest='trackid', nargs=1)
    args = parser.parse_args()
    trackid = int(args.trackid[0])
    main(trackid)
