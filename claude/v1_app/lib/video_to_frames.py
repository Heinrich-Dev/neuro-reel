"""
Extract 30 seconds of video into a folder of numbered JPEG frames using ffmpeg,
following CRCNS pvc-1 stimulus-frame conventions.

Usage:
    python3 video_to_frames.py INPUT_VIDEO [-o OUTPUT_DIR] [-d DURATION] [-r FPS]
                               [-s START] [-m MOVIE_ID] [-g SEGMENT_ID]
                               [--size WxH] [--quality Q]

Defaults:
    duration  30 s
    fps       30
    quality   2     (ffmpeg -qscale:v, 1=best ... 31=worst)
    naming    movieMMM_segSSS_frameFFFFF.jpg  (M=movie_id, S=segment_id, F=frame#)

Adjust the FRAME_PATTERN constant below to match your local pvc-1 stimulus folder
once you've confirmed the exact convention used in your ad1 download.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Adjust this to match the convention used in your pvc-1 stimulus folder.
# ffmpeg substitutes %0Nd for a zero-padded frame number; everything else
# in the string is taken literally. Python str.format placeholders {movie}
# and {segment} are filled in before handing the string to ffmpeg.
# ---------------------------------------------------------------------------
FRAME_PATTERN = 'movie{movie:03d}_seg{segment:03d}_frame%05d.jpg'


def check_ffmpeg():
    if shutil.which('ffmpeg') is None:
        sys.exit("ERROR: ffmpeg not found on PATH. Install it and retry.\n"
                 "  macOS:  brew install ffmpeg\n"
                 "  Ubuntu: sudo apt install ffmpeg\n"
                 "  Windows: https://ffmpeg.org/download.html")


def extract_frames(input_path, output_dir, duration=30.0, fps=30,
                   start=0.0, movie_id=0, segment_id=0,
                   size=None, quality=2, dry_run=False):
    """Run ffmpeg to extract `duration` seconds of `input_path` as JPEGs.

    Parameters
    ----------
    input_path : str | Path   -- source video file
    output_dir : str | Path   -- destination folder (created if missing, cleaned if not empty)
    duration   : float        -- clip length in seconds
    fps        : int          -- output frame rate (frames per second)
    start      : float        -- clip start offset in seconds
    movie_id   : int          -- value substituted for {movie} in the naming pattern
    segment_id : int          -- value substituted for {segment} in the naming pattern
    size       : str | None   -- optional 'WxH' to rescale frames (e.g. '320x240')
    quality    : int          -- ffmpeg -qscale:v JPEG quality (1=best, 31=worst)
    dry_run    : bool         -- if True, print the ffmpeg command and don't execute
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        sys.exit(f"ERROR: input video not found: {input_path}")

    # Prepare output directory
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Output directory '{output_dir}' is not empty — removing existing contents.")
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the ffmpeg frame-naming template
    name_template = FRAME_PATTERN.format(movie=movie_id, segment=segment_id)
    out_pattern = str(output_dir / name_template)

    # Build the ffmpeg command
    cmd = [
        'ffmpeg',
        '-hide_banner', '-loglevel', 'warning', '-stats',
        '-ss', str(start),               # seek BEFORE -i for fast keyframe seek
        '-i', str(input_path),
        '-t', str(duration),             # clip length
        '-vf', f'fps={fps}',             # resample to target fps
        '-qscale:v', str(quality),       # JPEG quality
        '-start_number', '0',            # frame numbering starts at 0
    ]
    if size:
        # Insert scale filter into the existing -vf
        cmd[cmd.index('-vf') + 1] = f'fps={fps},scale={size}'
    cmd.append(out_pattern)

    print("\nffmpeg command:")
    print("  " + " ".join(repr(c) if " " in c else c for c in cmd))

    if dry_run:
        print("\n(dry run — not executing)")
        return

    print(f"\nExtracting {duration}s at {fps} fps from '{input_path.name}' → '{output_dir}'")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nffmpeg exited with code {result.returncode}")

    # Summarize
    frames = sorted(output_dir.glob('*.jpg'))
    expected = int(round(duration * fps))
    print(f"\nDone. {len(frames)} frame(s) written (expected ~{expected}).")
    if frames:
        print(f"  First: {frames[0].name}")
        print(f"  Last:  {frames[-1].name}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a video clip into numbered JPEGs (pvc-1 style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', help='Path to input video file')
    parser.add_argument('-o', '--output', default='frames',
                        help='Output directory')
    parser.add_argument('-d', '--duration', type=float, default=30.0,
                        help='Clip length in seconds')
    parser.add_argument('-r', '--fps', type=int, default=30,
                        help='Output frame rate (frames per second)')
    parser.add_argument('-s', '--start', type=float, default=0.0,
                        help='Start offset in input video (seconds)')
    parser.add_argument('-m', '--movie-id', type=int, default=0,
                        help='movie_id value used in frame filenames')
    parser.add_argument('-g', '--segment-id', type=int, default=0,
                        help='segment_id value used in frame filenames')
    parser.add_argument('--size', default=None,
                        help='Resize frames, e.g. 320x240')
    parser.add_argument('--quality', type=int, default=2,
                        help='JPEG quality (ffmpeg -qscale:v, 1=best, 31=worst)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the ffmpeg command without running it')
    args = parser.parse_args()

    check_ffmpeg()
    extract_frames(
        input_path=args.input,
        output_dir=args.output,
        duration=args.duration,
        fps=args.fps,
        start=args.start,
        movie_id=args.movie_id,
        segment_id=args.segment_id,
        size=args.size,
        quality=args.quality,
        dry_run=args.dry_run,
    )


if __name__ == '__main__':
    main()
