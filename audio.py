import torch
import torchaudio


def load_audio(uri: str) -> torch.Tensor:
    """
    Takes in a uri for mono or stereo audio file, and returns tensor representation
    :param uri: uri of .wav audio file
    :return: tensor of shape (channels, time) of type torch.float32.
    """
    waveform, sample_rate = torchaudio.load(uri, normalize=True, channels_first=True)
    assert(sample_rate == 44100)
    # duplicate channel for mono
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return waveform
    

def framify(x: torch.Tensor, frame_size: int) -> torch.Tensor:
    """
    Splits up tensor into frames of size frame_size * 2.
    :param x: torch.Tensor of shape (channels, time) of type torch.float32
    :param frame_size: size of frames
    :return: torch.Tensor of shape (frames, features)
    """
    channels, time = x.shape
    assert(channels == 2)

    # slice (channels, time) -> (channels, frames, frame_size)
    frames = x.unfold(dimension=1, size=frame_size, step=frame_size)

    # rearrange to (frames, frame_size, channels)
    frames = frames.permute(1, 2, 0)

    # concatenate to create features [x1_left, x1_right, x2_left, x2_right, ...]
    frames = frames.reshape(frames.size(0), -1)

    return frames


def split_data(x: torch.Tensor, seq_length) -> torch.Tensor:
    """
    Takes in a tensor x of shape (frames, features) and splits it into M training examples.
    :return: tensor of shape (examples, seq_length, features)
    """
    # slice
    examples = x.unfold(dimension=0, size=seq_length, step=seq_length)
    examples = examples.permute(0, 2, 1)

    return examples