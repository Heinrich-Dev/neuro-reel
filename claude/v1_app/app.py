"""
Flask app: upload a 30 s video, extract frames, run the V1 model, return
predicted firing rates for visualization.

Endpoints:
    GET  /                       upload page
    POST /upload                 accept video, start background job, return job_id
    GET  /job/<job_id>           results page (polls for completion)
    GET  /api/job/<job_id>       JSON status + (when done) predictions
    GET  /static/baseline.json   precomputed per-electrode baseline + waveforms
"""
import os
import sys
import json
import threading
import subprocess
import traceback
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, render_template, abort, send_from_directory

import inference

# ---------------------------------------------------------------------------
APP_DIR    = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / 'uploads'
JOBS_DIR   = APP_DIR / 'jobs'
LIB_DIR    = APP_DIR / 'lib'

UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)

ALLOWED_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024   # 200 MB
TARGET_W, TARGET_H = 320, 240
TARGET_FPS = 30
TARGET_DURATION_S = 30

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_BYTES

# In-memory job registry. Proof-of-concept only; would not survive a restart.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _set_job(job_id, **updates):
    with JOBS_LOCK:
        JOBS[job_id].update(updates)


def _process_video(job_id, video_path, frames_dir):
    """Background worker: extract frames, run inference, save results."""
    try:
        _set_job(job_id, status='extracting_frames', progress=10,
                 message='Extracting 30 s of video as frames...')

        # Run video_to_frames.py as a subprocess. Force 320x240 to match the
        # model's expected input.
        script = LIB_DIR / 'video_to_frames.py'
        cmd = [
            sys.executable, str(script), str(video_path),
            '-o', str(frames_dir),
            '-d', str(TARGET_DURATION_S),
            '-r', str(TARGET_FPS),
            '--size', f'{TARGET_W}x{TARGET_H}',
            '--quality', '3',
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f'Frame extraction failed: {result.stderr[-500:]}')

        # Sanity-check
        n_frames = len(list(frames_dir.glob('*.jpg')))
        if n_frames < 50:
            raise RuntimeError(f'Only {n_frames} frames extracted; need a longer clip.')

        _set_job(job_id, status='running_model', progress=50,
                 message=f'Running V1 model on {n_frames} frames...')

        def on_progress(frac, msg):
            # Map inference progress [0, 1] into the 50–88% range of the overall job
            _set_job(job_id, progress=int(50 + frac * 38), message=msg)

        out = inference.run_inference(str(frames_dir), progress_cb=on_progress)

        _set_job(job_id, status='saving_results', progress=90,
                 message='Saving predictions...')

        # Persist a .npz for record-keeping; also serialize to JSON for the frontend.
        job_dir = JOBS_DIR / job_id
        np.savez(job_dir / 'predictions.npz',
                 predictions=out['predictions'],
                 rates_hz=out['rates_hz'],
                 times_s=out['times_s'])

        results_json = {
            'n_bins':         int(out['predictions'].shape[0]),
            'n_electrodes':   int(out['n_electrodes']),
            'bin_duration_s': float(out['bin_duration_s']),
            'times_s':        out['times_s'].tolist(),
            'rates_hz':       out['rates_hz'].tolist(),     # (n_bins, n_elec)
            'predictions':    out['predictions'].tolist(),
            'mean_rate_hz':   out['rates_hz'].mean(axis=0).tolist(),
            'n_frames_input': out['n_frames_input'],
        }
        with open(job_dir / 'results.json', 'w') as f:
            json.dump(results_json, f)

        _set_job(job_id, status='done', progress=100, message='Done',
                 result_path=str(job_dir / 'results.json'))
    except Exception as e:
        tb = traceback.format_exc()
        _set_job(job_id, status='error', progress=0,
                 message=str(e), traceback=tb)


# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html',
                           max_size_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
                           target_duration=TARGET_DURATION_S,
                           target_fps=TARGET_FPS)


@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'no file in upload'}), 400
    f = request.files['video']
    if not f.filename:
        return jsonify({'error': 'empty filename'}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({'error': f'unsupported file type {ext}. '
                        f'Allowed: {sorted(ALLOWED_EXTS)}'}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = job_dir / 'frames'
    video_path = job_dir / f'input{ext}'
    f.save(str(video_path))

    with JOBS_LOCK:
        JOBS[job_id] = {
            'status':   'queued',
            'progress': 0,
            'message':  'Queued',
            'filename': f.filename,
        }

    threading.Thread(
        target=_process_video,
        args=(job_id, video_path, frames_dir),
        daemon=True,
    ).start()

    return jsonify({'job_id': job_id})


@app.route('/job/<job_id>')
def job_page(job_id):
    with JOBS_LOCK:
        if job_id not in JOBS:
            abort(404)
    return render_template('results.html', job_id=job_id)


@app.route('/api/job/<job_id>')
def job_status(job_id):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if j is None:
            return jsonify({'error': 'unknown job'}), 404
        j = dict(j)   # shallow copy for response

    if j.get('status') == 'done':
        with open(JOBS_DIR / job_id / 'results.json') as f:
            j['results'] = json.load(f)
    return jsonify(j)


@app.route('/baseline')
def baseline():
    return send_from_directory(APP_DIR / 'static', 'baseline.json')


if __name__ == '__main__':
    print('Starting V1 predictor web app on http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
