
class Hyper_parameters():
    def __init__(self):
        self.dataset_path = "./dataset/gtzan"
        self.feature_path = "./dataset/feature_augment"
        self.genres = ['blues', 'classical', 'country', 'disco',
                       'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        # Feature Parameters
        self.sample_rate = 22050
        self.fft_size = 1024
        self.win_size = 1024
        self.hop_size = 512
        self.num_mels = 128
        self.feature_length = 1024

        # Training Parameters
        self.batch_size = 64
        self.num_epochs = 26
        self.learning_rate = 1e-4


HyperParams = Hyper_parameters()
