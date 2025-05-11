import glob
import mimetypes
import os
import platform
import shutil
import ssl
import urllib
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm
import subprocess
import roop.globals

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    try:
        subprocess.run(commands, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[❌] ffmpeg failed with return code {e.returncode}")
        return False


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30

def is_hwaccel_cuda_available() -> bool:
    try:
        output = subprocess.check_output(
            ['ffmpeg', '-hide_banner', '-hwaccels'],
            stderr=subprocess.STDOUT
        ).decode()
        return 'cuda' in output or 'cuvid' in output
    except Exception:
        return False

def can_use_cuda_hwaccel(target_path: str) -> bool:
    """Check if CUDA hwaccel is supported and usable for the given video."""
    if not is_hwaccel_cuda_available():
        return False
    try:
        subprocess.check_output(
            ['ffmpeg', '-hide_banner', '-loglevel', 'error',
             '-hwaccel', 'cuda', '-i', target_path, '-f', 'null', '-'],
            stderr=subprocess.STDOUT
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] CUDA hwaccel test failed: {e.output.decode(errors='ignore')}")
        return False

def extract_frames(target_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100
    output_pattern = os.path.join(temp_directory_path, f'%04d.{roop.globals.temp_frame_format}')

    ffmpeg_args = ['-i', target_path,
                   '-q:v', str(temp_frame_quality),
                   '-pix_fmt', 'rgb24',
                   '-vf', f'fps={fps}',
                   output_pattern]

    use_cuda = any(p.lower().startswith('cuda') for p in roop.globals.execution_providers)
    if use_cuda and can_use_cuda_hwaccel(target_path):
        print("[INFO] Using CUDA hardware decode")
        ffmpeg_args = ['-hwaccel', 'cuda'] + ffmpeg_args
    else:
        print("[INFO] Using CPU decode")

    return run_ffmpeg(ffmpeg_args)

def create_video(target_path: str, fps: float = 30) -> bool:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    output_video_quality = (roop.globals.output_video_quality + 1) * 51 // 100

    commands = [
        '-hwaccel', 'auto',
        '-framerate', str(fps),
        '-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format),
        '-c:v', roop.globals.output_video_encoder
    ]

    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])

    commands.extend([
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-y', temp_output_path
    ])

    return run_ffmpeg(commands)

def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format)))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Optional[str]:
    if source_path and target_path and output_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)

    # Handle broken symlink or file at temp dir path
    if os.path.exists(temp_directory_path) and not os.path.isdir(temp_directory_path):
        print(f"[WARN] Temp path exists but is not a directory. Removing: {temp_directory_path}")
        os.remove(temp_directory_path)

    # Try to clean and recreate the directory
    try:
        if os.path.isdir(temp_directory_path):
            print(f"[INFO] Removing stale temp directory: {temp_directory_path}")
            shutil.rmtree(temp_directory_path)

        Path(temp_directory_path).mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Created temp directory: {temp_directory_path}")
    except Exception as e:
        print(f"[ERROR] Could not create/reset temp directory: {e}")


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def rename_frames_sequentially(frame_paths: List[str]) -> List[str]:
    """Rename all frame image files to sequential format %08d.png."""
    if not frame_paths:
        return frame_paths

    directory = os.path.dirname(frame_paths[0])
    new_paths = []
    for idx, old_path in enumerate(sorted(frame_paths)):
        new_path = os.path.join(directory, f"{idx:08d}.png")
        if old_path != new_path:
            os.rename(old_path, new_path)
        new_paths.append(new_path)
    return new_paths

def compile_video_from_frames(frame_dir: str, output_video_path: str, fps: int = 30) -> None:
    print(f"[INFO] Compiling video from frames in {frame_dir} to {output_video_path}")

    # Ensure directory exists
    if not os.path.exists(frame_dir):
        print(f"[ERROR] Frame directory does not exist: {frame_dir}")
        return

    # Check PNGs exist
    frame_files = sorted(glob.glob(os.path.join(frame_dir, '*.png')))
    if not frame_files:
        print(f"[ERROR] No PNG frames found in directory: {frame_dir}")
        return

    # Format check
    expected_name = f"{0:08d}.png"
    if not os.path.basename(frame_files[0]).startswith("00000000"):
        print(f"[WARN] Frame filenames might not follow '%08d.png' format. Expected: {expected_name}")

    # Run ffmpeg
    command = [
        'ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(frame_dir, '%08d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_video_path
    ]
    try:
        subprocess.run(command, check=True)
        print("[INFO] Video compilation complete")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg failed: {e}")


def detect_audio_stream(video_path: str) -> bool:
    print(f"🔍 Audio stream check: {video_path}")
    command = ['ffprobe', '-i', video_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error']
    try:
        output = subprocess.check_output(command).decode().strip()
        return bool(output)
    except Exception:
        return False

def debug_audio_stream(path: str):
    print(f"🔍 Audio stream check: {path}")
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
             'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        codec = result.stdout.strip()
        if codec:
            print(f"[✅] Audio codec detected: {codec}")
        else:
            print(f"[⚠️] No audio codec detected.")
    except subprocess.CalledProcessError as e:
        print(f"[❌] ffprobe failed or no audio stream found.\n{e}")

def get_video_duration(path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[⚠️] Failed to get duration for {path}: {e}")
        return -1.0

def warn_if_duration_mismatch(video1: str, video2: str) -> None:
    dur1 = get_video_duration(video1)
    dur2 = get_video_duration(video2)
    if dur1 > 0 and dur2 > 0:
        delta = abs(dur1 - dur2)
        print(f"[DEBUG] Duration target: {dur1:.2f}s, output: {dur2:.2f}s (Δ {delta:.2f}s)")
        if delta > 0.2:
            print("[⚠️] WARNING: Output video duration differs from target! Audio sync issues may occur.")

def add_audio_to_video(output_path: str, target_video_path: str) -> None:
    print(f"[DEBUG] add_audio_to_video() called")
    print(f"[DEBUG] Output video path: {output_path}")
    print(f"[DEBUG] Target video (audio source) path: {target_video_path}")
    debug_audio_stream(target_video_path)

    print("[INFO] Adding original audio from target video...")
    temp_output_path = output_path.replace(".mp4", "_with_audio.mp4")
    print(f"[DEBUG] Temporary output path: {temp_output_path}")

    ffmpeg_cmd = [
        '-y',
        '-i', output_path,
        '-i', target_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        temp_output_path
    ]

    if not run_ffmpeg(ffmpeg_cmd):
        if getattr(roop.globals, "framewise", False):
            print("[INFO] Retrying with async audio resample to fix desync...")
            ffmpeg_cmd.insert(-2, '-af')
            ffmpeg_cmd.insert(-2, 'aresample=async=1')
            run_ffmpeg(ffmpeg_cmd)

    if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 0:
        debug_audio_stream(temp_output_path)
        os.replace(temp_output_path, output_path)
        print("[✅] Final swapped video generated with audio.")
        if getattr(roop.globals, "framewise", False):
            warn_if_duration_mismatch(target_video_path, output_path)
    else:
        print("[❌] Output with audio was not created correctly.")


def restore_audio(target_video_path: str, output_path: str) -> None:
    print(f"[DEBUG] restore_audio() called")
    print(f"[DEBUG] Target video path: {target_video_path}")
    print(f"[DEBUG] Output path: {output_path}")
    debug_audio_stream(target_video_path)

    if not detect_audio_stream(target_video_path):
        print("[WARN] Skipping audio restoration: no audio stream found.")
        return

    temp_output_path = output_path.replace(".mp4", "_with_audio.mp4")
    print(f"[DEBUG] Temporary output with audio: {temp_output_path}")

    ffmpeg_cmd = [
        '-i', output_path,
        '-i', target_video_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        '-y', temp_output_path
    ]

    if not run_ffmpeg(ffmpeg_cmd):
        if getattr(roop.globals, "framewise", False):
            print("[INFO] Retrying with async audio resample to fix desync...")
            ffmpeg_cmd.insert(-2, '-af')
            ffmpeg_cmd.insert(-2, 'aresample=async=1')
            run_ffmpeg(ffmpeg_cmd)

    if os.path.exists(temp_output_path):
        os.replace(temp_output_path, output_path)
        debug_audio_stream(output_path)
        if getattr(roop.globals, "framewise", False):
            warn_if_duration_mismatch(target_video_path, output_path)
        print("[✅] Audio restored successfully.")
    else:
        print("[❌] Failed to create audio output.")
