import torch
import random
import librosa
import numpy as np
import nlpaug.flow as naf
import nlpaug.augmenter.audio as naa
import nlpaug.augmenter.spectrogram as nas
from torchvision.transforms import Normalize

from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import Dataset
from nlpaug.augmenter.audio import AudioAugmenter
from src.datasets.root_paths import DATA_ROOTS

BAD_LIBRISPEECH_INDICES = [60150]
LIBRISPEECH_MEAN = [-22.924]
LIBRISPEECH_STDEV = [12.587]


class LibriSpeech(Dataset):

    def __init__(
            self,
            root=DATA_ROOTS['librispeech'],
            train=True,
            small=False,
            spectral_transforms=False,
            wavform_transforms=True,
            test_url='dev-clean',
            max_length=150526,
            input_size=224,
            normalize_mean=LIBRISPEECH_MEAN,
            normalize_stdev=LIBRISPEECH_STDEV,
        ):
        super().__init__()
        # choose to either apply augmentation at wavform or at augmentation level
        assert not (spectral_transforms and wavform_transforms)
        if train:
            if small:
                self.dataset = LIBRISPEECH(root, url='train-clean-100', download=True,
                                           folder_in_archive='LibriSpeech')
            else:
                self.dataset1 = LIBRISPEECH(root, url='train-clean-100', download=True,
                                            folder_in_archive='LibriSpeech')
                self.dataset2 = LIBRISPEECH(root, url='train-clean-360', download=True,
                                            folder_in_archive='LibriSpeech')
                self.dataset3 = LIBRISPEECH(root, url='train-other-500', download=True,
                                            folder_in_archive='LibriSpeech')
        else:
            self.dataset = LIBRISPEECH(root, url=test_url, download=True,
                                        folder_in_archive='LibriSpeech')

        self.spectral_transforms = spectral_transforms
        self.wavform_transforms = wavform_transforms
        self.max_length = max_length
        self.train = train
        self.small = small
        all_speaker_ids = self.get_speaker_ids()
        unique_speaker_ids = sorted(list(set(all_speaker_ids)))
        num_unique_speakers = len(unique_speaker_ids)
        self.speaker_id_map = dict(zip(unique_speaker_ids, range(num_unique_speakers)))
        self.all_speaker_ids = np.array([self.speaker_id_map[sid] for sid in all_speaker_ids])
        self.num_unique_speakers = num_unique_speakers
        self.num_labels = num_unique_speakers
        self.input_size = input_size
        self.FILTER_SIZE = input_size
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev

    def get_speaker_ids(self):
        if self.train and not self.small:
            speaker_ids_1 = self._get_speaker_ids(self.dataset1)
            speaker_ids_2 = self._get_speaker_ids(self.dataset2)
            speaker_ids_3 = self._get_speaker_ids(self.dataset3)
            return np.concatenate([speaker_ids_1, speaker_ids_2, speaker_ids_3])
        else:
            return self._get_speaker_ids(self.dataset)

    def _get_speaker_ids(self, dataset):
        speaker_ids = []
        for i in range(len(dataset)):
            fileid = dataset._walker[i]
            speaker_id = self.load_librispeech_speaker_id(
                fileid,
                dataset._path,
                dataset._ext_audio,
                dataset._ext_txt,
            )
            speaker_ids.append(speaker_id)
        return np.array(speaker_ids)

    def load_librispeech_speaker_id(self, fileid, path, ext_audio, ext_txt):
        speaker_id, _, _ = fileid.split("-")
        return int(speaker_id)

    def __getitem__(self, index):

        if self.train and not self.small:
            if index >= (len(self.dataset1) + len(self.dataset2)):
                try:
                    wavform, sample_rate, _, speaker_id, _, _ = \
                        self.dataset3.__getitem__(index - len(self.dataset1) - len(self.dataset2))
                except:
                    index2 = (index - len(self.dataset1) - len(self.dataset2) + 1) % len(self.dataset3)
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset3(index2)
            elif index >= len(self.dataset1):
                try:
                    wavform, sample_rate, _, speaker_id, _, _ = \
                        self.dataset2.__getitem__(index - len(self.dataset1))
                except:
                    index2 = (index - len(self.dataset1) + 1) % len(self.dataset2)
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset2.__getitem__(index2)
            else:
                try:
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset1.__getitem__(index)
                except:
                    index2 = (index + 1) % len(self.dataset)
                    wavform, sample_rate, _, speaker_id, _, _ = self.dataset1.__getitem__(index2)
        else:
            try:
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index)
            except:
                index2 = (index + 1) % len(self.dataset)
                wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index2)
                

        speaker_id = self.speaker_id_map[speaker_id]
        wavform = np.asarray(wavform[0])

        if self.wavform_transforms:
            transforms = WavformAugmentation(sample_rate)
            wavform = transforms(wavform)

        # pad to 150k frames
        if len(wavform) > self.max_length:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:self.max_length] if flip else 
                      wavform[-self.max_length:])
        else:
            padded = np.zeros(self.max_length)
            padded[:len(wavform)] = wavform  # pad w/ silence

        hop_length_dict = {224: 672, 112: 1344, 64: 2360, 32: 4800}
        spectrum = librosa.feature.melspectrogram(
            padded,
            sample_rate,
            hop_length=hop_length_dict[self.input_size],
            n_mels=self.input_size,
        )
        if self.spectral_transforms:  # apply time and frequency masks
            transforms = SpectrumAugmentation()
            spectrum = transforms(spectrum)

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        if self.spectral_transforms:  # apply noise on spectral
            noise_stdev = 0.25 * self.normalize_stdev[0]
            noise = torch.randn_like(spectrum) * noise_stdev
            spectrum = spectrum + noise

        normalize = Normalize(self.normalize_mean, self.normalize_stdev)
        spectrum = normalize(spectrum)

        return index, spectrum, speaker_id

    def __len__(self):
        if self.train and not self.small:
            return len(self.dataset1) + len(self.dataset2) + len(self.dataset3)
        else:
            return len(self.dataset)


class LibriSpeechTwoViews(LibriSpeech):

    def __getitem__(self, index):
        index, spectrum1, speaker_id = super().__getitem__(index)
        _, spectrum2, _ = super().__getitem__(index)

        return index, spectrum1, spectrum2, speaker_id


class LibriSpeechTransfer(Dataset):
    """
    Divide the dev-clean split of LibriSpeech into train and 
    test splits by speaker so we can train a logreg fairly.
    """
    def __init__(
            self,
            root=DATA_ROOTS['librispeech'],
            train=True,
            spectral_transforms=False,
            wavform_transforms=False,
            max_length=150526,
            input_size=224,
            normalize_mean=LIBRISPEECH_MEAN,
            normalize_stdev=LIBRISPEECH_STDEV,
        ):
        super().__init__()
        assert not (spectral_transforms and wavform_transforms)
        self.dataset = LIBRISPEECH(root, url='dev-clean', download=True,
                                  folder_in_archive='LibriSpeech')

        all_speaker_ids = self.get_speaker_ids(self.dataset)
        unique_speaker_ids = sorted(list(set(all_speaker_ids)))
        num_unique_speakers = len(unique_speaker_ids)
        self.speaker_id_map = dict(zip(unique_speaker_ids, range(num_unique_speakers)))
        self.all_speaker_ids = np.array([self.speaker_id_map[sid] for sid in all_speaker_ids])
        self.num_unique_speakers = num_unique_speakers
        self.num_labels = num_unique_speakers

        self.indices = self.train_test_split(self.dataset, all_speaker_ids, train=train)
        self.spectral_transforms = spectral_transforms
        self.wavform_transforms = wavform_transforms
        self.max_length = max_length
        self.train = train
        self.input_size = input_size
        self.normalize_mean = normalize_mean
        self.normalize_stdev = normalize_stdev

    def get_speaker_ids(self, dataset):
        speaker_ids = []
        for i in range(len(dataset)):
            fileid = dataset._walker[i]
            speaker_id = self.load_librispeech_speaker_id(
                fileid,
                dataset._path,
                dataset._ext_audio,
                dataset._ext_txt,
            )
            speaker_ids.append(speaker_id)
        return np.array(speaker_ids)

    def train_test_split(self, dataset, speaker_ids, train=True):
        rs = np.random.RandomState(42)  # fix seed so reproducible splitting

        unique_speaker_ids = sorted(set(speaker_ids))
        unique_speaker_ids = np.array(unique_speaker_ids)

        # train test split to ensure the 80/20 splits
        train_indices, test_indices = [], []
        for speaker_id in unique_speaker_ids:
            speaker_indices = np.where(speaker_ids == speaker_id)[0]
            size = len(speaker_indices)
            rs.shuffle(speaker_indices)
            train_size = int(0.8 * size)
            train_indices.extend(speaker_indices[:train_size].tolist())
            test_indices.extend(speaker_indices[train_size:].tolist())

        return train_indices if train else test_indices

    def load_librispeech_speaker_id(self, fileid, path, ext_audio, ext_txt):
        speaker_id, _, _ = fileid.split("-")
        return int(speaker_id)

    def __getitem__(self, index):
        # NOTE: overwrite index with our custom indices mapping exapmles
        #       to the training and test splits
        index = self.indices[index]

        try:
            wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index)
        except:
            index2 = (index + 1) % len(self.dataset)
            wavform, sample_rate, _, speaker_id, _, _ = self.dataset.__getitem__(index2)

        speaker_id = self.speaker_id_map[speaker_id]
        wavform = np.asarray(wavform[0])

        if self.wavform_transforms:
            transforms = WavformAugmentation(sample_rate)
            wavform = transforms(wavform)

        # pad to 150k frames
        if len(wavform) > self.max_length:
            # randomly pick which side to chop off (fix if validation)
            flip = (bool(random.getrandbits(1)) if self.train else True)
            padded = (wavform[:self.max_length] if flip else 
                      wavform[-self.max_length:])
        else:
            padded = np.zeros(self.max_length)
            padded[:len(wavform)] = wavform  # pad w/ silence

        hop_length_dict = {224: 672, 112: 1344, 64: 2360, 32: 4800}
        spectrum = librosa.feature.melspectrogram(
            padded,
            sample_rate,
            hop_length=hop_length_dict[self.input_size],
            n_mels=self.input_size,
        )

        if self.spectral_transforms:  # apply time and frequency masks
            transforms = SpectrumAugmentation()
            spectrum = transforms(spectrum)

        # log mel-spectrogram
        spectrum = librosa.power_to_db(spectrum**2)
        spectrum = torch.from_numpy(spectrum).float()
        spectrum = spectrum.unsqueeze(0)

        if self.spectral_transforms:  # apply noise on spectral
            noise_stdev = 0.25 * self.normalize_stdev[0]
            noise = torch.randn_like(spectrum) * noise_stdev
            spectrum = spectrum + noise

        normalize = Normalize(self.normalize_mean, self.normalize_stdev)
        spectrum = normalize(spectrum)

        return index, spectrum, speaker_id

    def __len__(self):
        return len(self.indices)


class SpectrumAugmentation(object):

    def get_random_freq_mask(self):
        return nas.FrequencyMaskingAug(mask_factor=40)
    
    def get_random_time_mask(self):
        return nas.TimeMaskingAug(mask_factor=40)

    def __call__(self, data):
        transforms = naf.Sequential([self.get_random_freq_mask(), 
                                     self.get_random_time_mask()])
        return transforms.augment(data)


class WavformAugmentation(object):

    def __init__(self, sample_rate=None, crop_and_noise_only=True):
        super().__init__()
        self.crop_and_noise_only = crop_and_noise_only
        self.sample_rate = sample_rate

    def get_random_loudness(self):
        return naa.LoudnessAug(crop=(0,1), coverage=1)

    def get_random_crop(self):
        return AudioCropAug(scale=(0.08, 1.0))

    def get_random_noise(self):
        return AudioNoiseAug(scale=1)

    def get_random_pitch(self):
        return naa.PitchAug(self.sample_rate, crop=(0,1), coverage=1)

    def __call__(self, data):
        if self.crop_and_noise_only:
            transforms = [self.get_random_crop(), self.get_random_noise()]
        else:
            transforms = [self.get_random_crop(), self.get_random_loudness(),
                          self.get_random_noise(), self.get_random_pitch()]
        random.shuffle(transforms)
        for transform in transforms:
            data = transform.augment(data)
        return data


class AudioCropAug(object):

    def __init__(self, scale=(0.08, 1.0), rescale=False):
        super().__init__()
        self.scale = scale
        self.rescale = rescale 
    
    def augment(self, data):
        scale = np.random.uniform(
            low=self.scale[0], 
            high=self.scale[1],
        )
        data_size = len(data)
        crop_size = int(scale * data_size)
        start_ix = int(np.random.choice(np.arange(data_size - crop_size)))
        crop = data[start_ix:start_ix+crop_size]

        if self.rescale:
            result = librosa.effects.time_stretch(crop, crop_size / data_size)
        else:
            result = np.zeros(data_size)
            result[start_ix:start_ix+crop_size] = crop

        return result


class AudioNoiseAug(object):

    def __init__(self, scale=1):
        super().__init__()
        self.scale = scale

    def get_noise(self, segment_size, scale):
        # https://en.wikipedia.org/wiki/Colors_of_noise
        uneven = segment_size % 2
        fft_size = segment_size // 2 + 1 + uneven
        noise_fft = np.random.randn(fft_size)
        noise_fft = noise_fft * scale  # magnify?
        noise = np.fft.irfft(noise_fft)
        if uneven:
            noise = noise[:-1]
        return noise

    def augment(self, data):
        noise = self.get_noise(len(data), self.scale)
        return data + noise
