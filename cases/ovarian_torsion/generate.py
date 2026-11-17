"""Generate the ovarian torsion case audio, one line at a time, joined with pauses.

Usage (needs a GPU and HF_TOKEN with IndicF5 gating accepted):
    python cases/ovarian_torsion/generate.py
Output: cases/ovarian_torsion/ovarian_torsion.wav (24 kHz mono)
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from inference import load_model, synthesize  # noqa: E402

SR = 24_000
REF_WAV = ROOT / "data/reference_audio/hindi_ref.wav"
REF_TXT = (ROOT / "data/reference_audio/hindi_ref.txt").read_text(encoding="utf-8").strip()
# The "..." gaps in the original prompt become longer pauses before key reveals.
PAUSE_S = {"Yeh kya hai?": 1.2, "Ovarian torsion.": 0.9}
DEFAULT_PAUSE_S = 0.6


def main() -> None:
    lines = [l.strip() for l in (HERE / "script.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    model = load_model()
    chunks = []
    for line in lines:
        print(f"-> {line}")
        audio = synthesize(model, line, str(REF_WAV), REF_TXT)
        chunks.append(audio.astype(np.float32))
        chunks.append(np.zeros(int(SR * PAUSE_S.get(line, DEFAULT_PAUSE_S)), dtype=np.float32))
    out = np.concatenate(chunks)
    out = out / max(1e-6, float(np.abs(out).max())) * 0.9
    dst = HERE / "ovarian_torsion.wav"
    sf.write(dst, out, SR)
    print(f"wrote {dst} ({len(out) / SR:.1f}s)")


if __name__ == "__main__":
    main()
