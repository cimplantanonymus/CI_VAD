# Code for INTERSPEECH 2026 Submission

This repository contains the code to reproduce the experiments for our paper: *"Less is More: Cochlear Implant Strategies as Robust Feature Extractors for Low Resource Voice Activity Classification."*

Due to the double-blind review process, all author names, affiliations, and identifying metadata have been scrubbed from this repository.

## Requirements
* Python 3.10+
* `torch`, `torchaudio`, `torch_directml`
* `numpy`, `librosa`, `soundfile`, `scipy`, `scikit-learn`, `tqdm`, `jupyter`

### Dataset Preparation
Due to licensing restrictions (LDC) and file size constraints, the raw datasets used in this study are not included in this repository. To run this code, you must obtain the following datasets independently:

1. **DARPA TIMIT:** Obtain from the Linguistic Data Consortium (LDC).
2. **Noise Datasets:** Obtain the DEMAND dataset and NOISEX-92.

Make sure you download 16kHz versions of each noise type from DEMAND.

#### Required Directory Structure
Create a folder named `TIMIT` in the same directory as the `vaddesign.ipynb` notebook. Organize the downloaded audio files exactly as follows before running the code. 

*(Note: The test script dynamically reads all subfolders placed inside `TEST_NOISE/`. You can place your chosen unseen noise environments there).*

```text
project_root/
│
├── vaddesign.ipynb
└── TIMIT/
    ├── data/
    │   ├── TRAIN/   # Place TIMIT DR1-DR8 folders here (must contain .WAV and .PHN files)
    │   └── TEST/    # Place TIMIT DR1-DR8 folders here (must contain .WAV and .PHN files)
    │
    ├── TRAIN_NOISE/ # Seen training noises (names must match exactly)
    │   ├── BABBLE/babble.wav
    │   ├── DKITCHEN/ch01.wav
    │   ├── DWASHING/ch01.wav
    │   ├── OOFFICE/ch01.wav
    │   ├── PCAFETER/ch01.wav
    │   ├── STRAFFIC/ch01.wav
    │   └── TBUS/ch01.wav
    │
    └── TEST_NOISE/  # Unseen testing noises (dynamically loaded)
        ├── DLIVING/ch01.wav
        ├── NFIELD/ch01.wav
        ├── NPARK/ch01.wav
        ├── NRIVER/ch01.wav
        ├── OHALLWAY/ch01.wav
        ├── OMEETING/ch01.wav
        ├── PRESTO/ch01.wav
        ├── PSTATION/ch01.wav
        ├── SPSQUARE/ch01.wav
        ├── TCAR/ch01.wav
        └── TMETRO/ch01.wav
```
##### How to Run the Code

1. Launch Jupyter Notebook in your terminal from the project root directory:
   $ jupyter notebook

2. Open `vaddesign.ipynb` in your browser.
3. In the top menu bar, click "Kernel" -> "Restart & Run All".

###### Notebook Cell Information

The notebook is designed to run sequentially:

Cell 1 (VAD Label Generation): 
- Parses the TIMIT .PHN files to generate frame-level VAD labels (0/1) for speech vs. non-speech at a 10ms frame hop.

Cell 2 (Noisy Mixture Creation): 
- Mixes the clean TIMIT audio with the structured noise datasets at specified SNR levels (-5 dB, 0 dB, 5 dB).

Cell 3 (Feature Extraction): 
- Extracts both wideband audio features and simulates the specific envelope feature strategies for three Cochlear Implant (CI) manufacturers: Cochlear (ACE), MED-EL (CIS), and Advanced Bionics (AB).

Cell 4 (Wideband TCN Training): 
- Trains the baseline Temporal Convolutional Network (TCN) directly on the wideband acoustic features.

Cell 5 (Industry CI TCN Training): 
- Iteratively trains a dedicated TCN model for each CI manufacturer's extracted features.

Cell 6 (Standard Evaluation): 
- Evaluates all trained models against the test dataset and generates a standard metrics report.

Cell 7 (Detailed/Comparative Evaluation): 
- Generates a more comprehensive evaluation and compares the TCN outputs against the pre-trained Silero VAD baseline.

###### Outputs

During execution, the script will automatically generate the following data directories within the `./TIMIT/` folder:
- /manifests
- /frame_labels
- /noisy_audio
- /noisy_manifests
- /processed_features_wideband
- /industry_features

Model Checkpoints Saved (in root directory):
- best_wideband_tcn_512.pth
- best_cochlear_tcn_512.pth
- best_medel_tcn_512.pth
- best_ab_tcn_512.pth

Evaluation Metrics Saved (in root directory):
- vad_test_metrics.csv
- vad_test_metrics_detailed.csv