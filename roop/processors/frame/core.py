import os
import sys
import importlib
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm
import cv2
from roop.utilities import detect_fps, is_video, restore_audio, detect_audio_stream
from roop.utilities import compile_video_from_frames
from roop.globals import output_path
import roop
from roop.utilities import add_audio_to_video

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    'pre_check',
    'pre_start',
    'process_frame',
    'process_frames',
    'process_image',
    'process_video',
    'post_process'
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(f'roop.processors.frame.{frame_processor}')
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                raise NotImplementedError
    except ModuleNotFoundError:
        sys.exit(f'Frame processor {frame_processor} not found.')
    except NotImplementedError:
        sys.exit(f'Frame processor {frame_processor} not implemented correctly.')
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    return FRAME_PROCESSORS_MODULES


def multi_process_frame(source_path: str, temp_frame_paths: List[str], process_frames: Callable[[str, List[str], Any], None], update: Callable[[], None]) -> None:
    with ThreadPoolExecutor(max_workers=roop.globals.execution_threads) as executor:
        futures = []
        queue = create_queue(temp_frame_paths)
        queue_per_future = max(len(temp_frame_paths) // roop.globals.execution_threads, 1)
        while not queue.empty():
            future = executor.submit(process_frames, source_path, pick_queue(queue, queue_per_future), update)
            futures.append(future)
        for future in as_completed(futures):
            future.result()


def create_queue(temp_frame_paths: List[str]) -> Queue[str]:
    queue: Queue[str] = Queue()
    for frame_path in temp_frame_paths:
        queue.put(frame_path)
    return queue


def pick_queue(queue: Queue[str], queue_per_future: int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


# def process_video(source_path: str, frame_paths: list[str], process_frames: Callable[[str, List[str], Any], None]) -> None:
#     progress_bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
#     total = len(frame_paths)
#     with tqdm(total=total, desc='Processing', unit='frame', dynamic_ncols=True, bar_format=progress_bar_format) as progress:
#         multi_process_frame(source_path, frame_paths, process_frames, lambda: update_progress(progress))


def core_process_video(source_path: str, target_path: str, frame_paths: List[str], process: Callable, is_framewise: bool = False):
    directory = os.path.dirname(frame_paths[0])
    for idx, old_path in enumerate(sorted(frame_paths)):
        new_path = os.path.join(directory, f"{idx:08d}.png")
        if old_path != new_path:
            os.rename(old_path, new_path)
        frame_paths[idx] = new_path

    with tqdm(total=len(frame_paths), desc='Processing', unit='frame') as progress:
        if is_framewise:
            for path in frame_paths:
                frame = cv2.imread(path)
                result = process(source_path, target_path, frame)
                cv2.imwrite(path, result)
                progress.update(1)
        else:
            process(source_path, frame_paths, lambda: update_progress(progress))


    fps = detect_fps(source_path)
    compile_video_from_frames(directory, roop.globals.output_path, fps)

    if is_video(roop.globals.target_path) and detect_audio_stream(roop.globals.target_path):
        add_audio_to_video(roop.globals.output_path, roop.globals.target_path)
    else:
        print("[WARN] Skipping audio restoration: no audio stream found.")


def update_progress(progress: Any = None) -> None:
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / 1024 / 1024 / 1024
    progress.set_postfix({
        'memory_usage': '{:.2f}'.format(memory_usage).zfill(5) + 'GB',
        'execution_providers': roop.globals.execution_providers,
        'execution_threads': roop.globals.execution_threads
    })
    progress.refresh()
    progress.update(1)
