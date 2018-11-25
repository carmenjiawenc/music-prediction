import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pyaudio
import numpy as np
from tqdm import tqdm
from util import compute_features, get_model, pca, CATEGORIES

RATE = 44100
RECORD_SECONDS = 30
CHUNKSIZE = 2048

pa = pyaudio.PyAudio()
print("Machine SampleRate: %s" % pa.get_device_info_by_index(0)['defaultSampleRate'])
stream = pa.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=CHUNKSIZE)
model = get_model()

print("* Start Recording for %d seconds" % RECORD_SECONDS)
stream.start_stream()
frames = []

for i in tqdm(range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS))):
    data = stream.read(CHUNKSIZE, exception_on_overflow=False)
    frames.append(np.fromstring(data, dtype=np.float32))

print("* End Recording")
stream.stop_stream()
stream.close()
pa.terminate()

audio_data = np.hstack(frames)
features = compute_features('test', audio_data, RATE)
features = pca(features)
pred = model.predict(features)
cat = np.argmax(pred)
# print(pred)
print(CATEGORIES[cat])