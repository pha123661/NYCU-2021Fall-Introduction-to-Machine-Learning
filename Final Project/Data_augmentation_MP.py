import librosa
import numpy as np
import os
import soundfile as sf
from Hyper_parameters import HyperParams
from multiprocessing import Pool

def get_item(genre):
    return librosa.util.find_files(HyperParams.dataset_path + '/' + str(genre))


def readfile(file_name):
    y, sr = librosa.load(file_name, HyperParams.sample_rate)
    return y, sr


def change_pitch_and_speed(data):
    y_pitch_speed = data.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1)
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed


def change_pitch(data, sr):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def value_aug(data):
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    y_aug = y_aug * dyn_change
    return y_aug


def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise


def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
    return y_harmonic, y_percussive


def shift(data):
    return np.roll(data, 1600)


def stretch(data, rate=1):
    input_length = len(data)
    streching = librosa.effects.time_stretch(data, rate)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching

def change_speed(data):
    y_speed = data.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def MP_helper(in_):
    y, idx = in_
    if idx == 0:
        ret = add_noise(y)
    elif idx == 1:
        ret = shift(y)
    elif idx == 2:
        ret = stretch(y)
    elif idx == 3:
        ret = change_pitch_and_speed(y)
    elif idx == 4:
        ret = change_pitch(y, HyperParams.sample_rate)
    elif idx == 5:
        ret = change_speed(y)
    elif idx == 6:
        ret = value_aug(y)
    elif idx == 7:
        _, ret = hpss(y)
    elif idx == 8:
        ret = shift(y)
    return ret

def main():
    print('Augmentation')
    genres = HyperParams.genres

    for genre in genres:
        item_list = get_item(genre)
        for file_name in item_list:
            y, sr = readfile(file_name)
            with Pool(12) as pool:
                [data_noise, data_roll, data_stretch, pitch_speed, pitch, speed, value, y_percussive, y_shift] \
                    = pool.map(MP_helper, [(y, x) for x in range(9)])
            print([x.shape for x in [data_noise, data_roll, data_stretch, pitch_speed, pitch, speed, value, y_percussive, y_shift]])
            save_path = os.path.join(file_name.split(genre + '.')[0])
            save_name =  genre + '.'+file_name.split(genre + '.')[1]
            print(save_name)

            sf.write(os.path.join(save_path, save_name.replace('.wav', 'a.wav')), data_noise, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'b.wav')), data_roll, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'c.wav')), data_stretch, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'd.wav')), pitch_speed, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'e.wav')), pitch, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'f.wav')), speed, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'g.wav')), value, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'h.wav')), y_percussive, HyperParams.sample_rate)
            sf.write(os.path.join(save_path, save_name.replace('.wav', 'i.wav')), y_shift, HyperParams.sample_rate)
        print('finished')


if __name__ == '__main__':
    main()