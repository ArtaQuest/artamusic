# ArtaMusic

Generating a finished record — song, master, cover — on **free hardware**, with every claim
measured rather than asserted. This is the production pipeline behind **UNBROKEN**, an epic
hybrid-trap anthem submitted through [ArtaQuest](https://artaquest.com)'s reproducibility gate
(work 9320). The publication run is [`arash0ash/unbroken-final`](https://www.kaggle.com/code/arash0ash/unbroken-final):
one public notebook that generates the song by cover-conditioning on the previously published male
take, GATES each candidate in-run (register on a demucs-isolated stem + Whisper word accuracy,
with the measuring code fetched from this repo pinned by commit), masters with a static gain and
4x-oversampled true-peak limiter, and renders the cover — so the published files are provably the
run's own outputs, and the claims are checked in the run that made them. Gate verdict of record:
seed 4242 ACCEPTED, 138.2 Hz male, 87.3% word accuracy; master -10.2 LUFS, LRA 5.6 LU against the
reference record's 5.4.

## Layout

| path | what |
|------|------|
| `stages/song_xl.py` | the song stage — ACE-Step XL 4.6B on a free Kaggle P100, with an offload ladder |
| `lib/measure.py` | audio measurement: pitch (YIN), tempo, loudness/LRA, register — `selftest` first |
| `lib/lyric_profile.py` | scores a lyric against the measured craft profile of the reference record |
| `lib/intelligibility.py` | Whisper word-error-rate of a sung take against the lyric it was given |
| `lib/kpush.py` | Kaggle push/poll (`--acc` does nothing; see findings) |
| `validation/xl_probe.py` | the probe that established the 4.6B model fits a 16 GB P100 |
| `validation/pins.json` | every model pinned by commit sha — a tag can move, a sha cannot |
| `validation/craft_targets.json` | lyric craft targets, measured with the same instrument that scores |
| `song/lyrics_unbroken.txt` | the lyric as sung |

**The 4-line outro is the lyric's ending, not a truncation** — it is the final station of the
forge spine (heat → hammer → quench → edge): verse 1, verse 2, bridge and outro each close on
their "call it …" line, in that order, and every chorus is word-identical. Both are locked
invariants: `lyric_profile.py` fails loudly if any edit breaks the spine order or the chorus
identity, so the gate trips automatically instead of relying on a note being read.

The craft reference is Glum Aleks' *Typical Story*. Its text is **not** included here (it is not
ours to redistribute); `craft_targets.json` carries the measured profile, and the profiler
recomputes targets from any local copy so instrument and target can never drift apart.

## Findings that were expensive to learn

- **The 4.6B model fits a 16 GB P100 — peak 12.12 GB, no offloading.** An OOM had been read as
  "too big" and the pipeline dropped to a 1.1B model. The error actually reported 2.30 GiB
  reserved-but-unallocated and suggested the fix itself: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
  The ceiling was allocator fragmentation, one environment variable deep.
- **Throughput, not memory, is the real XL constraint.** The model loads with 3.7 GB spare, then
  needs ~1000 s to render 180 s of audio — past ACE-Step's default 600 s kill. Six takes died at
  that wall before `ACESTEP_GENERATION_TIMEOUT` (a documented env var) was raised.
- **Kaggle always schedules a Tesla P100 (sm_60) and the accelerator cannot be pinned** —
  `machine_shape` round-trips to `"Gpu"` across every spelling tried. Kaggle's own preinstalled
  torch has no sm_60 kernels; install `torch==2.7.1+cu126` *before the first `import torch`*.
- **float16 overflows to NaN in the 4.6B DiT on sm_60** (`nan=280000`, four seeds). bfloat16 has
  float32's exponent range and runs at 0.81x float32 speed on this card — hardware, not emulation.
  `bitsandbytes` needs sm_75 and is unavailable.
- **Vocal register is a lottery no caption controls.** Across 12 takes with male-explicit captions
  (both model sizes), one came back male. Register must be measured on a **demucs-isolated stem**
  with YIN — mix-based estimators lock onto the 808's harmonics and returned exact FFT-bin
  multiples, wrong on 8 of 8 takes.
- **ffmpeg `loudnorm` silently discards `linear=true`** when the target is unreachable and runs a
  gain rider instead — it collapsed loudness range 7.1 → 3.0 LU against a 5.4 LU reference. Use a
  static gain plus a 4x-oversampled limiter (`alimiter` at base rate is sample-peak, not true-peak).
- **Cross-correlation cannot verify sync on this material** — an 808's period matches any sensible
  search window, and it manufactured three phantom desyncs. Residual energy at lag 0 is unambiguous.
  Two decoders of the same MP3 differ by 1,258 samples of codec offset; naive comparisons inherit it.
- **Every estimator that was not validated against known-truth signals turned out to be wrong.**
  A tempo estimator read 128 BPM as 64.00; a formant estimator missed /i/'s F2 by 1,600 Hz; a peak
  meter scraped ffmpeg text, defaulted plausibly, and shipped a clipped file. `measure.py selftest`
  is the entry point, not an afterthought.

## Licence

Code is MIT. The lyric under `song/` is © the ArtaQuest Foundation, all rights reserved.
