# HRAudioWizard CLI

A high-resolution audio batch processor for WAV files, supporting remastering, noise reduction, high frequency compensation, and more. This tool is a command-line interface (CLI) version of the original HRAudioWizard, designed for easy and interactive use without a GUI. I've also removed the license feature to make it easier to use.

## Features
- Batch processing of WAV files
- Remastering (bass/treble enhancement)
- Noise reduction (spectral gating)
- High frequency compensation (HFC)
- Compressed source mode
- Adjustable output bit depth (24, 32, 64)
- Adjustable sampling rate scale (x1, x2, x4, x8)
- Customizable lowpass filter
- License file support
- Interactive mode (step-by-step prompts)
- Progress bars for each file (via tqdm)

## Requirements
- Python 3.8+
- numpy
- scipy
- librosa
- pyaudio (requires PortAudio development headers)
- soundfile
- tqdm

### Linux: Install PortAudio development headers

Before installing Python dependencies, you must install the PortAudio development package:

```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
```

Then install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode
Run the script with no arguments to enter interactive mode:

```bash
python HRAudioWizard.py
```

You will be prompted for:
- Input WAV file paths (comma-separated)
- Output bit depth (24, 32, 64)
- Sampling rate scale (x1, x2, x4, x8)
- Remastering (y/n)
- Noise reduction (y/n)
- High frequency compensation (y/n)
- Compressed source mode (y/n)
- Lowpass filter frequency (Hz)
- License file path (optional)

### Command-Line Mode
You can also specify all options directly:

```bash
python HRAudioWizard.py input1.wav input2.wav \
  --bit-depth 24 \
  --scale 2 \
  --remaster \
  --noise-reduction \
  --high-freq-comp \
  --compressed-mode \
  --lowpass 20000 \
  --license /path/to/license.lc_hraw
```

#### Arguments
- `files` (positional): Input WAV files to process (one or more)
- `--bit-depth`: Output bit depth (24, 32, 64). Default: 24
- `--scale`: Sampling rate scale (1, 2, 4, 8). Default: 1
- `--remaster`: Enable remastering
- `--noise-reduction`: Enable noise reduction
- `--high-freq-comp`: Enable high frequency compensation
- `--compressed-mode`: Enable compressed source mode
- `--lowpass`: Lowpass filter frequency in Hz. Default: 16000

### Output
- Each input file will be processed and saved as `<originalname>_converted.wav` in the same directory.
- Progress bars are shown for each file.

## License
- The tool requires a valid license file for full functionality. Place your license file as `license.lc_hraw` in the script directory, or specify its path when prompted.
- Without a license, the tool may run in limited or trial mode.

## Notes
- Only WAV files are supported as input.
- For best results, use high-quality stereo WAV files.
- The tool is designed for batch/offline processing. Real-time audio features are not included in the CLI version.

## Troubleshooting
- If you encounter errors related to missing libraries, ensure all dependencies are installed.
- For licensing issues, check your license file and its path.
- For audio processing errors, verify your input files are valid stereo WAV files.

## Credits
- Original author: [Super-YH](https://github.com/Super-YH)
- CLI adaptation by the open-source community.

---

For more details, see the source code in `HRAudioWizard.py`.
