import unittest
import torch

import torchaudio
import os

from audio import load_audio, framify, split_data


class TestAudio(unittest.TestCase):

    def setUp(self):
        self.sample_rate = 44100
        self.duration_seconds = 1
        self.frame_size = 32
        self.seq_length = 10

        # dummy data
        self.mono_waveform = torch.randn(1, self.sample_rate)
        self.stereo_waveform = torch.randn(2, self.sample_rate)

        # save fake audio
        self.test_file_stereo = "test_file_stereo.wav"
        self.test_file_mono = "test_file_mono.wav"
        torchaudio.save(self.test_file_stereo, self.stereo_waveform, self.sample_rate)
        torchaudio.save(self.test_file_mono, self.mono_waveform, self.sample_rate)

    def tearDown(self):
        if os.path.exists(self.test_file_stereo):
            os.remove(self.test_file_stereo)
        if os.path.exists(self.test_file_mono):
            os.remove(self.test_file_mono)

    def test_load_audio_stereo(self):
        waveform = load_audio(self.test_file_stereo)
        self.assertEqual(
            waveform.shape, self.stereo_waveform.shape,
            "Loaded waveform shape mismatch"
        )
        self.assertTrue(
            torch.allclose(waveform, self.stereo_waveform, atol=1e-5),
            "Waveform content mismatch"
        )

    def test_load_audio_mono(self):
        waveform = load_audio(self.test_file_mono)
        self.assertEqual(
            waveform.shape, (2, self.mono_waveform.size(1)),
            "Stereo waveform shape mismatch"
        )
        self.assertTrue(
            torch.allclose(waveform[0], self.mono_waveform[0]),
            "Stereo channels are not identical"
        )
        self.assertTrue(
            torch.allclose(waveform[1], self.mono_waveform[0]),
            "Stereo channels are not identical"
        )

    def test_framify(self):
        frames = framify(self.stereo_waveform, self.frame_size)
        expected_frames = self.stereo_waveform.size(1) // self.frame_size
        self.assertEqual(
            frames.shape, (expected_frames, self.frame_size * 2),
            "Framify output shape mismatch"
        )

        left_frames = self.stereo_waveform[0]
        right_frames = self.stereo_waveform[1]
        for i, frame in enumerate(frames):
            for j in range(0, len(frame)//2):
                self.assertTrue(
                    torch.allclose(frame[j*2], left_frames[i * self.frame_size + j]),
                    "Left channel feature mismatch"
                )
                self.assertTrue(
                    torch.allclose(frame[j*2+1], right_frames[i * self.frame_size + j]),
                    "Right channel feature mismatch"
                )

    def test_split_data(self):
        frames = framify(self.stereo_waveform, self.frame_size)
        training_data = split_data(frames, self.seq_length)

        expected_examples = frames.size(0) // self.seq_length
        self.assertEqual(
            training_data.shape,
            (expected_examples, self.seq_length, frames.shape[1]),
            "Split data output shape mismatch"
        )

    def test_load_audio_input(self):
        waveform = load_audio("./data/test/ht1-input.wav")
        self.assertEqual(
            waveform.size(0), 2,
            "Stereo waveform shape mismatch"
        )


if __name__ == "__main__":
    unittest.main()
