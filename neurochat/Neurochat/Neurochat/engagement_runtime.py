import threading
from eeg_streamer import run_stream
from eeg_engagement import EngagementScorer
from eeg_engagement import make_synthetic_eeg

# Global singleton
scorer = EngagementScorer(sample_rate=256, n_channels=8)

# simple calibration using synthetic data (replace later with real calibration process)
scorer.calibrate(
    make_synthetic_eeg(120, 256, 8, "low"),
    make_synthetic_eeg(120, 256, 8, "high"),
)

def get_score():
    return scorer.get_score_for_prompt()

def freeze():
    return scorer.freeze()

def unfreeze():
    scorer.unfreeze()

# start background EEG stream
threading.Thread(target=run_stream, args=(scorer,), daemon=True).start()