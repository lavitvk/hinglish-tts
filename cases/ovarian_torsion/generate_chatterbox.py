"""Generate the ovarian torsion case audio with Chatterbox Multilingual (Hindi). Runs on CPU.

Setup (Python 3.11):
    uv venv -p 3.11 cbenv && . cbenv/bin/activate
    uv pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match chatterbox-tts "setuptools<81" faster-whisper
Run:
    python cases/ovarian_torsion/generate_chatterbox.py [--ref voice.wav]

Each line is generated up to --tries times. A take is rejected if it runs too
long for its text (Chatterbox sometimes babbles past the end) or if Whisper
can't read it back. The best take per line is kept, trimmed, and joined with pauses.
Outputs: ovarian_torsion.wav, lines/NN.wav, qc.tsv
"""
import argparse
import difflib
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

HERE = Path(__file__).resolve().parent
PAUSE_S = {"यह क्या है?": 1.4, "ओवेरियन टॉर्शन।": 0.9}  # thinking gap before the reveal
DEFAULT_PAUSE_S = 0.6


def trim(wav: np.ndarray, sr: int, thr: float = 0.01, pad: float = 0.08) -> np.ndarray:
    f = int(sr * 0.02)
    e = np.array([np.sqrt((wav[i:i + f] ** 2).mean()) for i in range(0, len(wav) - f, f)])
    idx = np.where(e > thr)[0]
    if not len(idx):
        return wav
    a, b = max(0, idx[0] * f - int(pad * sr)), min(len(wav), (idx[-1] + 1) * f + int(pad * sr))
    return wav[a:b]


def norm(s: str) -> str:
    return re.sub(r"[\s।?,.!]", "", s).replace("़", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="optional voice-clone reference wav (~10 s)")
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--only", help="comma-separated line numbers to redo; other lines reuse lines/NN.wav")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxMultilingualTTS.from_pretrained(device=device)
    try:
        from faster_whisper import WhisperModel
        asr = WhisperModel("small", device=device, compute_type="int8" if device == "cpu" else "float16")
    except ImportError:
        asr = None
    sr = tts.sr
    lines = [l.strip() for l in (HERE / "script_devanagari.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    (HERE / "lines").mkdir(exist_ok=True)

    only = {int(x) for x in args.only.split(",")} if args.only else None
    qc_path = HERE / "qc.tsv"
    old_qc = qc_path.read_text(encoding="utf-8").splitlines()[1:] if qc_path.exists() else []
    out, qc = [], ["line\ttake\tdur_s\tmax_s\tasr_score\tasr_text"]
    for n, line in enumerate(lines, 1):
        if only is not None and n not in only and (HERE / "lines" / f"{n:02d}.wav").exists():
            wav, _ = sf.read(HERE / "lines" / f"{n:02d}.wav", dtype="float32")
            qc.append(old_qc[n - 1] if n - 1 < len(old_qc) else line)
            out += [wav, np.zeros(int(sr * PAUSE_S.get(line, DEFAULT_PAUSE_S)), dtype=np.float32)]
            continue
        max_s = 0.11 * len(norm(line)) + 0.9
        best = None
        for take in range(args.tries):
            torch.manual_seed(1000 * n + take + 100 * (only is not None))
            wav = trim(tts.generate(line, language_id="hi", audio_prompt_path=args.ref,
                                    exaggeration=0.5, cfg_weight=0.5,
                                    temperature=args.temperature).squeeze(0).cpu().numpy(), sr)
            dur = len(wav) / sr
            heard, score = "", 1.0
            if asr is not None:
                tmp = HERE / "lines" / "_tmp.wav"
                sf.write(tmp, wav, sr)
                segs, _ = asr.transcribe(str(tmp), language="hi", beam_size=5)
                heard = "".join(s.text for s in segs).strip()
                tmp.unlink()
                score = difflib.SequenceMatcher(None, norm(line), norm(heard)).ratio()
            ok = dur <= max_s
            rank = (ok, score)
            print(f"[{n:02d}.{take}] {dur:4.1f}s/{max_s:3.1f} asr={score:.2f} {'OK ' if ok else 'LONG'} {line} | {heard}", flush=True)
            if best is None or rank > best[0]:
                best = (rank, take, wav, dur, max_s, score, heard)
            if ok and score >= 0.75:
                break
        _, take, wav, dur, max_s, score, heard = best
        sf.write(HERE / "lines" / f"{n:02d}.wav", wav, sr)
        qc.append(f"{line}\t{take}\t{dur:.2f}\t{max_s:.2f}\t{score:.2f}\t{heard}")
        out += [wav, np.zeros(int(sr * PAUSE_S.get(line, DEFAULT_PAUSE_S)), dtype=np.float32)]

    full = np.concatenate([np.zeros(int(sr * 0.3), dtype=np.float32)] + out)
    full = full / max(1e-6, float(np.abs(full).max())) * 0.9
    sf.write(HERE / "ovarian_torsion.wav", full, sr)
    (HERE / "qc.tsv").write_text("\n".join(qc) + "\n", encoding="utf-8")
    print(f"wrote ovarian_torsion.wav ({len(full) / sr:.1f}s)")


if __name__ == "__main__":
    main()
