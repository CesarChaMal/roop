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
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
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


def extract_frames(target_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(target_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100
    return run_ffmpeg(['-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)])


def create_video(target_path: str, fps: float = 30) -> bool:
    temp_output_path = get_temp_output_path(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    output_video_quality = (roop.globals.output_video_quality + 1) * 51 // 100
    commands = ['-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format), '-c:v', roop.globals.output_video_encoder]
    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])
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
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


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

def add_audio_to_video(output_path: str, target_video_path: str) -> None:
    print("[INFO] Adding original audio from target video...")

    temp_output_path = output_path.replace(".mp4", "_with_audio.mp4")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", output_path,           # video (swapped frames)
        "-i", target_video_path,     # audio (original audio)
        "-c:v", "copy",
        "-c:a", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        temp_output_path
    ], check=True)

    os.replace(temp_output_path, output_path)

    print("[✅] Final swapped video generated with audio.")

