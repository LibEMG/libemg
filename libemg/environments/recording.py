"""Record what happens in an environment, in the form an OfflineDataHandler reads.

An environment knows what it showed and what it was told: the model's output,
where the cursor and the target were, which trial it was, and -- because it
knows the goal -- what the user *should* have been doing at that moment, which
is what context-informed incremental learning (CIIL) trains on. It does not see
the EMG. The online data handler does. So a run is recorded in two halves and
joined afterwards.

What a run saves
----------------
::

    <folder>/
        _session_emg.csv            every EMG sample the online data handler
                                    received: arrival time, then the channels.
                                    The same file continuous screen guided
                                    training writes (other modalities the
                                    handler streams are logged beside it).
        _session_environment.csv    one row per frame the environment drew:
                                    its time, its trial, and the fields it
                                    recorded, under a header.
        recording.json              what was recorded, by whom, and how to read
                                    it back.
        trial_<n>.csv               the training files. One row per EMG sample
                                    produced during trial n, carrying the state
                                    of the frame it was produced in.

The training files are plain numeric CSVs with a header row, so they need
nothing special to read. :func:`load_recording` is the one-line route into an
:class:`~libemg.data_handler.OfflineDataHandler`::

    from libemg.environments.recording import load_recording

    odh = load_recording('data/fitts_2026-10-28_14-03-11')
    windows, metadata = odh.parse_windows(80, 16, metadata_operations={'ciil_class': 'last_sample'})
    # metadata['ciil_class'], metadata['optimal_x'], metadata['prediction'], ...

and it is no more than this call, which works just as well written by hand::

    odh.get_data('data/fitts_...', [RegexFilter('trial_', '.csv', trials, '')],
                 metadata_fetchers=[ColumnFetcher('ciil_class', 12), ...],
                 data_column=[1, 2, ..., 8], skiprows=1)

Joining the halves
------------------
The EMG file stamps each sample with the time its *batch* was read, which is
later than the sample was produced and the same for a whole burst. The sample
clock is recovered with :func:`libemg.session.fit_production_clock`, the fit
continuous screen guided training uses to cut its repetitions, and every sample
is then given the most recent frame drawn at or before the moment it was
produced. Both clocks are ``time.time()`` on the same machine.

The join is a pure function of the files, like cutting a session: if something
goes wrong at the end of a run -- the window is closed mid-write, say -- the raw
halves are still there and :func:`assemble` rebuilds the training files from
them::

    python -c "from libemg.environments.recording import assemble; assemble('data/fitts_...')"

What each field means
---------------------
Fields come in groups, and :class:`RecordingConfig` chooses which are kept.

``timestamp``
    When the sample was produced (or, without EMG, when the frame was drawn),
    in ``time.time()`` seconds. The raw ``_session_`` files always carry their
    own times, since nothing could be joined without them.
``model``
    What the environment received from the model and acted on, held from one
    message until the next: ``prediction`` (the class) and ``speed`` (the
    proportional control value) from a classifier, ``output_x`` / ``output_y``
    from a regressor.
``game``
    The environment's own state. For the cursor tasks: ``cursor_x``,
    ``cursor_y``, ``target_x``, ``target_y`` and ``target_radius`` in screen
    pixels (y grows downwards, as pygame draws), ``in_target``, and the
    cursor's movement this frame.
``adaptation``
    What the user should have been doing. ``optimal_x`` / ``optimal_y`` is the
    unit vector from the cursor to the target in the model's output space, zero
    inside the target. ``ciil_class`` is the class a classifier should have
    output, by the CIIL rule in :func:`libemg.adaptation._base.ciil_direction`,
    and ``ciil_x`` / ``ciil_y`` the tolerant CIIL label a regressor should
    have (:func:`libemg.adaptation._base.produce_tciil_feedback`).

``trial`` is always kept. It is what the training files are split by.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

#: The two halves, and the description of both.
FRAMES_NAME = "_session_environment.csv"
EMG_PREFIX = "_session_"
EMG_NAME = EMG_PREFIX + "emg.csv"
MANIFEST_NAME = "recording.json"
#: A training file, one per trial.
TRIAL_TEMPLATE = "trial_{trial}.csv"

#: Field groups, as :class:`RecordingConfig` names them.
MODEL = "model"
GAME = "game"
ADAPTATION = "adaptation"
GROUPS = (MODEL, GAME, ADAPTATION)


@dataclass(frozen=True)
class RecordingConfig:
    """What to save from a run of an environment, and where.

    Parameters
    ----------
    folder: str
        Where this run's files go. It is created if it does not exist. Use a
        fresh folder per run: the EMG log appends to what is already there.
    timestamps: bool, default=True
        Put a ``timestamp`` column in the training files.
    emg: bool, default=True
        Record the EMG. Needs an online data handler to read it from; without
        one, the training files hold one row per frame instead.
    model_outputs: bool, default=True
        Record what the model sent: the class and speed, or the regressor's
        outputs.
    game: bool, default=True
        Record the environment's state: cursor, target, whether the cursor is
        in it, and so on.
    adaptation: bool, default=True
        Record what the user should have been doing: the optimal direction and
        the CIIL label.
    """

    folder: str
    timestamps: bool = True
    emg: bool = True
    model_outputs: bool = True
    game: bool = True
    adaptation: bool = True

    def keeps(self, group):
        """Whether fields of ``group`` are saved."""
        return {MODEL: self.model_outputs, GAME: self.game,
                ADAPTATION: self.adaptation}.get(group, True)


class FrameLog:
    """The environment's half: one row per frame, built up as the game runs.

    Columns are not declared in advance. A field is added the first frame it
    is recorded in, and a frame that did not record it reads as NaN, so an
    environment records what it has -- a regressor's two outputs, a
    classifier's class -- without listing it anywhere else first.
    """

    def __init__(self):
        self.groups = {}        # column -> group, in the order first seen
        self.rows = []

    def add(self, timestamp, trial, model=None, game=None, adaptation=None):
        """Record one frame.

        Parameters
        ----------
        timestamp: float
            ``time.time()`` when the frame was drawn.
        trial: int
            The trial the frame belongs to.
        model, game, adaptation: dict or None
            Field name to value. A value may be a number, a bool, None (read
            as NaN), or a sequence, which is spread over ``<name>_0``,
            ``<name>_1``, ...
        """
        row = {"time": float(timestamp), "trial": float(trial)}
        for group, fields in ((MODEL, model), (GAME, game), (ADAPTATION, adaptation)):
            for name, value in (fields or {}).items():
                for column, number in _flatten(name, value):
                    self.groups.setdefault(column, group)
                    row[column] = number
        self.rows.append(row)

    def save(self, config, environment, details=None):
        """Write ``_session_environment.csv`` and ``recording.json`` into ``config.folder``.

        Parameters
        ----------
        config: RecordingConfig
            Which groups to keep.
        environment: str
            The environment's name, for the manifest.
        details: dict or None
            Anything else worth knowing later: the class each action was
            mapped to, the environment's settings. Written as JSON, with
            anything JSON cannot hold written as text.
        """
        folder = Path(config.folder)
        folder.mkdir(parents=True, exist_ok=True)
        kept = [column for column, group in self.groups.items() if config.keeps(group)]
        columns = ["time", "trial"] + kept
        table = np.array([[row.get(column, np.nan) for column in columns]
                          for row in self.rows], dtype=float).reshape(-1, len(columns))
        # repr-precision floats: a timestamp needs its microseconds and an EMG
        # sample its five significant figures, and one fixed format can't serve both.
        np.savetxt(folder / FRAMES_NAME, table, delimiter=",", fmt="%.17g",
                   header=",".join(columns), comments="")
        manifest = {
            "schema": SCHEMA_VERSION,
            "environment": environment,
            "created": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(config),
            "details": details or {},
            "frames": {"file": FRAMES_NAME, "count": len(self.rows),
                       "columns": {column: self.groups.get(column, "structure")
                                   for column in columns}},
        }
        _write_manifest(folder, manifest)
        return folder / FRAMES_NAME


class EnvironmentRecording:
    """The EMG half of a run, and the join once the run is over.

    Start it before the environment and stop it after; stopping assembles the
    training files. Use it as a context manager in a script::

        config = RecordingConfig(folder='data/fitts_1')
        env = Fitts(controller, FittsConfig(num_trials=20), recording=config)
        with EnvironmentRecording(config, online_data_handler=odh):
            env.run()
        odh_offline = load_recording(config.folder)

    The GUI does the same around a task launched with a save folder.

    Parameters
    ----------
    config: RecordingConfig
        What to save and where. ``config.emg`` decides whether EMG is logged.
    online_data_handler: OnlineDataHandler or None
        Where the EMG comes from. Without one the run is recorded frame by
        frame only.
    """

    def __init__(self, config, online_data_handler=None):
        self.config = config
        self.online_data_handler = online_data_handler
        self.logging = False

    @property
    def folder(self):
        return Path(self.config.folder)

    @property
    def records_emg(self):
        return bool(self.config.emg and self.online_data_handler is not None)

    def start(self):
        """Create the folder and start logging EMG, if asked to and able to."""
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.records_emg:
            # The thread logger, not log_to_file: it is reading by the time this
            # returns, so the recording starts before the environment's first frame.
            self.online_data_handler.start_log(str(self.folder / EMG_PREFIX))
            self.logging = True
        return self

    def stop(self, assemble_files=True):
        """Stop logging and, by default, build the training files.

        Returns
        ----------
        list of Path
            The training files written; empty if the environment recorded no
            frames (it was stopped before its first, or it was never told to
            record).
        """
        if self.logging:
            self.online_data_handler.stop_log()
            self.logging = False
        if not assemble_files:
            return []
        return assemble(self.folder)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


def assemble(folder):
    """Join a run's EMG and frames into one training file per trial.

    Safe to run again: it reads only the ``_session_`` files and the manifest,
    and overwrites the training files it writes.

    Parameters
    ----------
    folder: str or Path
        A run's folder.

    Returns
    ----------
    list of Path
        The training files, in trial order.
    """
    import pandas as pd
    from libemg.session import fit_production_clock

    folder = Path(folder)
    manifest = _read_manifest(folder)
    frames_path = folder / FRAMES_NAME
    if manifest is None or not frames_path.exists():
        return []
    config = RecordingConfig(**manifest["config"])
    frames = pd.read_csv(frames_path)
    for stale in folder.glob(TRIAL_TEMPLATE.format(trial="*")):
        stale.unlink()
    if frames.empty:
        manifest["trials"] = []
        _write_manifest(folder, manifest)
        return []

    frame_times = frames["time"].to_numpy(dtype=float)
    fields = frames.drop(columns=["time"])

    emg_path = folder / EMG_NAME
    emg_info = None
    if config.emg and emg_path.exists() and emg_path.stat().st_size > 0:
        raw = pd.read_csv(emg_path, header=None).to_numpy(dtype=float)
        arrivals, samples = raw[:, 0], raw[:, 1:]
        fit = fit_production_clock(arrivals)
        if fit is not None:
            fs, anchor, spread = fit
            produced = anchor + np.arange(len(samples)) / fs
            emg_info = {"file": EMG_NAME, "fs": fs, "anchor": anchor, "lag_spread": spread}
        else:
            # Too short to fit a clock to. Arrival times are late by the
            # link's latency, but they are the best there is.
            produced = arrivals
            emg_info = {"file": EMG_NAME, "fs": None, "anchor": None, "lag_spread": None}
        emg_info["samples"] = int(len(samples))
        # The frame on screen when each sample was produced. Samples from
        # before the first frame or after the last one belong to no trial.
        frame_of = np.searchsorted(frame_times, produced, side="right") - 1
        period = float(np.median(np.diff(frame_times))) if len(frame_times) > 1 else 0.0
        inside = (frame_of >= 0) & (produced <= frame_times[-1] + period)
        frame_of, produced, samples = frame_of[inside], produced[inside], samples[inside]
        table = pd.DataFrame(samples, columns=[f"emg_{c}" for c in range(samples.shape[1])])
        table = pd.concat([table, fields.iloc[frame_of].reset_index(drop=True)], axis=1)
        times = produced
    else:
        table = fields.reset_index(drop=True)
        times = frame_times
    if config.timestamps:
        table.insert(0, "timestamp", times)

    written, trials = [], []
    for trial, rows in table.groupby("trial", sort=True):
        trial = int(trial)
        path = folder / TRIAL_TEMPLATE.format(trial=trial)
        rows.to_csv(path, index=False)
        written.append(path)
        trials.append({"trial": trial, "file": path.name, "rows": int(len(rows))})

    columns = list(table.columns)
    manifest["emg"] = emg_info
    manifest["trials"] = trials
    manifest["trial_files"] = {
        "template": TRIAL_TEMPLATE,
        "rows": "emg samples" if emg_info else "frames",
        "columns": columns,
        "data_columns": [i for i, c in enumerate(columns) if c.startswith("emg_")],
        "skiprows": 1,
    }
    _write_manifest(folder, manifest)
    return written


def load_recording(folder, fields=None):
    """Read a run's training files into an OfflineDataHandler.

    ``data`` holds the EMG, one array per trial. Every other column becomes
    per-sample metadata under its own name -- ``trial``, ``timestamp``,
    ``prediction``, ``ciil_class``, ``optimal_x``, ``cursor_x`` and so on --
    ready for :meth:`OfflineDataHandler.parse_windows`. Combine runs with
    ``+``.

    Parameters
    ----------
    folder: str or Path
        A run's folder, after :func:`assemble`.
    fields: list of str or None
        Only these metadata columns. Defaults to all of them.

    Returns
    ----------
    OfflineDataHandler
    """
    from libemg.data_handler import ColumnFetcher, OfflineDataHandler, RegexFilter

    folder = Path(folder)
    manifest = _read_manifest(folder)
    if manifest is None or "trial_files" not in manifest:
        raise FileNotFoundError(f"{folder} has no assembled recording. Run "
                                "libemg.environments.recording.assemble on it first.")
    described = manifest["trial_files"]
    if not described["data_columns"]:
        raise ValueError(f"{folder} was recorded without EMG, so there is nothing to train "
                         f"on; its frames are in {FRAMES_NAME}.")
    columns = described["columns"]
    wanted = [c for c in columns if not c.startswith("emg_")
              and (fields is None or c in fields)]
    trials = [str(entry["trial"]) for entry in manifest["trials"]]
    odh = OfflineDataHandler()
    # The filter only picks out the trial files; the trial number itself comes
    # from its column, which is why the filter stores nothing.
    odh.get_data(str(folder), [RegexFilter("trial_", ".csv", trials, "")],
                 metadata_fetchers=[ColumnFetcher(c, columns.index(c)) for c in wanted],
                 data_column=described["data_columns"], skiprows=described["skiprows"])
    return odh


def _flatten(name, value):
    if value is None:
        return [(name, np.nan)]
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float).ravel()
        return [(f"{name}_{i}", float(v)) for i, v in enumerate(values)]
    number = float(value)
    return [(name, number if math.isfinite(number) else np.nan)]


def _read_manifest(folder):
    path = Path(folder) / MANIFEST_NAME
    if not path.exists():
        return None
    with open(path) as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema", SCHEMA_VERSION)) > SCHEMA_VERSION:
        raise ValueError(f"{path} was written by a newer LibEMG.")
    return manifest


def _write_manifest(folder, manifest):
    path = Path(folder) / MANIFEST_NAME
    scratch = path.with_suffix(".json.part")
    with open(scratch, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    scratch.replace(path)
    return path
