#!/bin/bash
# Full command history: ZEUS (unsupervised PC) + ZETA (text-based) experiments on AudioEditingCode.
# Run on: vast.ai RTX 5070 Ti box, ssh -p 30677 root@116.127.115.43
# Not meant to be run top-to-bottom blindly -- extraction .pt paths below are literal outputs
# from earlier steps (contain timestamps), and env setup only needs to happen once.

## repo setup
cd /workspace
git clone -b codeclean https://github.com/SairajLoke/AudioEditingCode.git   # cloned user's fork
cd AudioEditingCode
uv venv --python 3.12 .venv                                                # venv, python 3.12.3
source .venv/bin/activate

## dependency install + fixes (repo is 2024-era, needed pinning against today's ecosystem)
uv pip install -r requirements.txt                                         # base install
uv pip install setuptools                                                  # pkg_resources missing otherwise
uv pip install soxr                                                        # missing transitive dep, broke diffusers import chain
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchaudio torchvision --force-reinstall  # wrong: no Blackwell/sm_120 kernels
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio torchvision --force-reinstall  # correct: matches RTX 5070 Ti + driver ceiling
uv pip install --index-url https://download.pytorch.org/whl/cu128 torchcodec --force-reinstall  # torchaudio.load needs this now, must match torch's cu tag
uv pip install 'transformers==4.44.2' --reinstall                          # unpinned default (5.16.1) broke encode_text API
uv pip install 'diffusers==0.30.0' --reinstall                             # unpinned default (0.40.0) needs newer transformers than above
uv pip install 'huggingface-hub<1.0' --reinstall                           # diffusers==0.30.0 requires this explicitly
uv pip install 'numpy<2.0.0' --reinstall                                   # kept getting bumped to 2.x by the above reinstalls, repo needs <2.0
uv pip freeze > /workspace/AudioEditingCode/pip_freeze_working_env.txt     # environment snapshot

## repo patch
# code/main_pc_apply_drift.py line 31: default evals_pt "./eigvals.pt" (file never shipped in repo) -> None
# makes apply use the extraction's own per-timestep eigenvalues instead of a missing averaged reference
sed -i "31s|.*|    parser.add_argument('--evals_pt', type=str, default=None, help=\"Use precomputed eigvalues\")  # own evals|" code/main_pc_apply_drift.py

## input prep
mkdir -p inputs && cd inputs
curl -sL 'https://raw.githubusercontent.com/HilaManor/AudioEditingCode/codeclean/docs/resources/audio/orig/MDDBGospel_8secs.mp3' -o MDDBGospel_8secs.mp3   # site's demo clip
curl -sL 'https://raw.githubusercontent.com/HilaManor/AudioEditingCode/codeclean/docs/resources/audio/orig/MDDBGospel.mp3' -o MDDBGospel_full.mp3          # full untrimmed track, 77.65s
ffmpeg -y -i MDDBGospel_8secs.mp3 -ar 16000 -ac 1 -sample_fmt s16 MDDBGospel_8secs.wav   # mp3->wav: audioldm.utils.get_duration needs real RIFF wav
ffmpeg -y -i MDDBGospel_full.mp3 -ar 16000 -ac 1 -sample_fmt s16 MDDBGospel_full.wav
ffmpeg -y -i MDDBGospel_full.wav -t 16 -ar 16000 -ac 1 -sample_fmt s16 MDDBGospel_16sec.wav   # first 16s, to fit the 5-10s mask seen in the paper figure
cd ..

export WANDB_MODE=disabled     # scripts call wandb.login() unconditionally; avoids hang on headless box
export HF_HOME=/workspace/.hf_home
cd code

## sanity check before committing to a full ~26min extraction
CUDA_VISIBLE_DEVICES=0 python main_pc_extract_inv.py \
  --init_aud /workspace/AudioEditingCode/inputs/MDDBGospel_8secs.wav \
  --model_id cvssp/audioldm2-music \
  --results_path /workspace/AudioEditingCode/pc_extractions \
  --drift_start 150 --drift_end -1 --n_evs 3 --dry     # --dry: pipeline load + full loop, no PC extraction, fast

## RUN 1: 8sec clip, unmasked, matches website's exact demo audio
CUDA_VISIBLE_DEVICES=0 python main_pc_extract_inv.py \
  --init_aud /workspace/AudioEditingCode/inputs/MDDBGospel_8secs.wav \
  --model_id cvssp/audioldm2-music \
  --source_prompt "A recording of ryhtmic clapping, a women singing, and drums and guitar playing." \
  --results_path /workspace/AudioEditingCode/pc_extractions \
  --drift_start 150 --drift_end -1 --n_evs 3
# -> pc_extractions/audioldm2-music/MDDBGospel_8secs/pmt_.../sNone_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788367761.pt

EXT1="/workspace/AudioEditingCode/pc_extractions/audioldm2-music/MDDBGospel_8secs/pmt_A_recording_of_ryhtmic_clapping,_a_women_singing,_and_drums_and_guitar_playing.__neg__/sNone_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788367761.pt"
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT1" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --amount 2    # PC3, +2
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT1" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --amount -2   # PC3, -2
# finding: PC3 here tracks instrumental/percussive balance, not vibrato (see run1 config.json)

## RUN 2: 16sec clip, masked 5-10s (per paper fig 4b mask bracket), PC3
CUDA_VISIBLE_DEVICES=0 python main_pc_extract_inv.py \
  --init_aud /workspace/AudioEditingCode/inputs/MDDBGospel_16sec.wav \
  --model_id cvssp/audioldm2-music \
  --source_prompt "A recording of ryhtmic clapping, a women singing, and drums and guitar playing." \
  --results_path /workspace/AudioEditingCode/pc_extractions_masked \
  --drift_start 150 --drift_end -1 --n_evs 3 --patch 128 256    # patch = 5s,10s at ~25.625 latent-frames/sec

EXT2="/workspace/AudioEditingCode/pc_extractions_masked/audioldm2-music/MDDBGospel_16sec/pmt_A_recording_of_ryhtmic_clapping,_a_women_singing,_and_drums_and_guitar_playing.__neg__/sNone_p128-256_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788369572.pt"
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT2" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --amount 2     # no fix_alpha -> leaks outside mask
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT2" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --amount -2
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT2" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --amount 2 --fix_alpha 0.025    # corrected: paper's App.D data-leakage fix
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT2" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --amount -2 --fix_alpha 0.025
# finding: masked+fixalpha PC3 behaves like a real vibrato axis (F0 stable register, oscillation depth moves as paper's figure claims)

## RUN 2b: diagnostic sweep, user reported voice "breaking" inside the mask -- test strength and PC choice
for CFG in "1 1.5" "1 -1.5" "2 1.5" "2 -1.5" "3 0.5" "3 -0.5" "3 1" "3 -1" "3 1.5" "3 -1.5"; do
  set -- $CFG; EV=$1; AMT=$2
  CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT2" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs $EV --fix_alpha 0.025 --amount $AMT
done
# finding: deviation-from-baseline scales cleanly with amount, but roughness proxy stays flat -- not a pure strength issue

## RUN 3: 16sec clip, masked 3-7s (user requested alternate window), PC3
CUDA_VISIBLE_DEVICES=0 python main_pc_extract_inv.py \
  --init_aud /workspace/AudioEditingCode/inputs/MDDBGospel_16sec.wav \
  --model_id cvssp/audioldm2-music \
  --source_prompt "A recording of ryhtmic clapping, a women singing, and drums and guitar playing." \
  --results_path /workspace/AudioEditingCode/pc_extractions_masked \
  --drift_start 150 --drift_end -1 --n_evs 3 --patch 77 179    # patch = 3s,7s

EXT3="/workspace/AudioEditingCode/pc_extractions_masked/audioldm2-music/MDDBGospel_16sec/pmt_A_recording_of_ryhtmic_clapping,_a_women_singing,_and_drums_and_guitar_playing.__neg__/sNone_p77-179_pc-both_cfgd3_drift150--1_it50_c1.0e-03_1788374316.pt"
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT3" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --fix_alpha 0.025 --amount 2
CUDA_VISIBLE_DEVICES=0 python main_pc_apply_drift.py --extraction_path "$EXT3" --drift_start 150 --drift_end -1 --use_specific_ts_pc 120 --evs 3 --fix_alpha 0.025 --amount -2

## RUN 4: ZETA (text-based, no PCA) alternative -- fully deterministic given fixed prompts
CUDA_VISIBLE_DEVICES=0 python main_run.py \
  --init_aud /workspace/AudioEditingCode/inputs/MDDBGospel_16sec.wav \
  --model_id cvssp/audioldm2-music \
  --source_prompt "A recording of ryhtmic clapping, a women singing, and drums and guitar playing." \
  --target_prompt "A recording of ryhtmic clapping, a women singing with a strong vocal vibrato, and drums and guitar playing." \
  --tstart 100 \
  --results_path /workspace/AudioEditingCode/results_zeta_vibrato
# note: main_run.py builds save path via string concat not os.path.join -- actual output landed at
# code/workspace/AudioEditingCode/results_zeta_vibrato/... despite the absolute --results_path given

## analysis scripts (local python, not repo code -- librosa-based DSP checks on the outputs above)
python analyze_vibrato.py       # F0/vibrato-band/harmonic-percussive metrics, run1 outputs
python analyze_vibrato_run2.py  # same, run2 outputs, + masked-window-only variants
python check_leakage.py         # RMS(a2-a-2) inside vs outside mask, before/after fix_alpha
python analyze_sweep.py         # same metrics across the full run2b PC/amount sweep
