import os
import numpy as np
import librosa

from Hyper_parameters import HyperParams

def load_list(list_name, HyperParams):
    with open(os.path.join(HyperParams.dataset_path, list_name)) as f:
        file_names = f.read().splitlines()
        return file_names

def melspectrogram(file_name, HyperParams):
    y, sr = librosa.load(file_name, HyperParams.sample_rate)
    S = librosa.stft(y, n_fft=HyperParams.fft_size, hop_length=HyperParams.hop_size, win_length=HyperParams.win_size)

    mel_basis = librosa.filters.mel(HyperParams.sample_rate, n_fft=HyperParams.fft_size, n_mels=HyperParams.num_mels)
    mel_S = np.dot(mel_basis, np.abs(S))
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.T

    return mel_S

def resize_array(array, length):
    resized_array = np.zeros((length, array.shape[1]))
    if array.shape[0] >= length:
        resize_array = array[:length]
    else:
        resized_array[:array.shape[0]] = array
    return resize_array

def main():
    print("Extracting Feature")

    for root, _, filenames in os.walk(HyperParams.dataset_path):
        for file_name in filenames:
            file_name = os.path.join(root, file_name)
            feature = melspectrogram(file_name, HyperParams)
            feature = resize_array(feature, HyperParams.feature_length)
            # Data Arguments
            num_chunks = feature.shape[0]/HyperParams.num_mels
            data_chuncks = np.split(feature, num_chunks)
            for idx, i in enumerate(data_chuncks):
                save_path = os.path.join(HyperParams.feature_path, file_name.split('\\')[1])
                save_name = file_name.split('\\')[1]+"."+file_name.split(".")[2]+"."+str(idx)+".npy"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                np.save(os.path.join(save_path, save_name), i.astype(np.float32))
                print(os.path.join(save_path, save_name))


    print('finished')

if __name__ == '__main__':
	main()