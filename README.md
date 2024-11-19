# GuitarTransformer
Exploring Transformer Architecture in Guitar Amplifier Emulators

Data borrowed from https://github.com/Alec-Wright/Automated-GuitarAmpModelling

# Testing
To run all tests in /test starting with 'test_'
python3 -m unittest discover -s test

To run specific file
python3 -m unittest test.test_audio

To run specific test case
python3 -m unittest test.test_audio.TestAudio.test_framify