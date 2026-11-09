"""Directory sequencing tied to completion of both DA3 passes, not prompt count."""
import hashlib
from dataclasses import dataclass
from pathlib import Path

DIRECTORY_KEY = '__duke_da3_directory__'
EXTENSIONS = {'.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v'}


def idle_guard():
    yield


@dataclass
class DirectoryRun:
    files: tuple
    directory: str
    owner: str
    index: int = 0
    identities: tuple = ()

    def __post_init__(self):
        self.identities = tuple(self.identity(p) for p in self.files)

    @staticmethod
    def identity(p):
        stat = p.stat()
        identity = f'{p.resolve()}:{stat.st_size}:{stat.st_mtime_ns}'
        return hashlib.sha256(identity.encode()).hexdigest()[:24]

    @property
    def source_key(self):
        key = self.identity(self.files[self.index])
        if key != self.identities[self.index]:
            raise ValueError('Source video changed during the directory run')
        return key


def directory_run(manager):
    entry = manager.outputs.get(DIRECTORY_KEY) if manager is not None else None
    return entry[0] if entry else None


def rewind(manager):
    """Reopen VHS readers while preserving the pinned directory selection."""
    entry = manager.outputs.pop(DIRECTORY_KEY, None)
    manager.reset()
    if entry:
        manager.outputs[DIRECTORY_KEY] = entry


def finish_video(manager):
    """Called only after video closure, successful audio mux and cache cleanup."""
    run = directory_run(manager)
    if run is None:
        if not manager.outputs:
            manager.reset()
        return False
    unexpected = set(manager.outputs) - {DIRECTORY_KEY}
    if unexpected:
        raise ValueError('DA3 directory mode needs a dedicated VHS Batch Manager')
    entry = manager.outputs.pop(DIRECTORY_KEY)
    manager.reset()
    run.index += 1
    if run.index == len(run.files):
        entry[-1].close()
        return False
    manager.outputs[DIRECTORY_KEY] = entry
    return True


class DukeDA3DirectoryQueue:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'directory': ('STRING', {'default': '', 'tooltip': 'Input directory; source list is fixed for this run.'}),
            'meta_batch': ('VHS_BatchManager',),
            'start_index': ('INT', {'default': 0, 'min': 0}),
            'max_videos': ('INT', {'default': 0, 'min': 0, 'tooltip': '0 = all remaining videos'})},
            'hidden': {'unique_id': 'UNIQUE_ID'}}

    RETURN_TYPES = ('STRING', 'STRING', 'INT', 'INT')
    RETURN_NAMES = ('video_path', 'source_key', 'current_index', 'video_count')
    FUNCTION = 'execute'
    CATEGORY = 'Stereo/DA3'

    def execute(self, directory, meta_batch, start_index=0, max_videos=0, unique_id=None):
        root = Path(directory).expanduser().resolve()
        run = directory_run(meta_batch)
        if run is None:
            if not root.is_dir():
                raise ValueError('DA3 input directory does not exist')
            files = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
                           key=lambda p: p.name)
            files = files[start_index:]
            if max_videos:
                files = files[:max_videos]
            if not files:
                raise ValueError('No videos at the selected directory start index')
            run = DirectoryRun(tuple(files), str(root), str(unique_id))
            guard = idle_guard()
            next(guard)
            meta_batch.outputs[DIRECTORY_KEY] = (run, guard)
        elif run.directory != str(root) or run.owner != str(unique_id):
            raise ValueError('Directory selection changed during a DA3 run')
        return (str(run.files[run.index]), run.source_key, run.index + 1, len(run.files))
