"""One choreography run, from Start to the files it leaves behind.

:class:`ChoreographyController` owns a scheduler and everything that hangs off
it, and nothing that draws: the GUI calls :meth:`~ChoreographyController.tick`
once per frame and renders whatever pose :meth:`~ChoreographyController.pose`
returns; tests call the same methods with a fake clock.

Two modes, never mixed:

``preview``
    Animates the schedule. Nothing is recorded, no annotation file is written,
    and no device is needed.
``recording``
    Opens the collection service's continuous recording (through a
    :class:`CollectionLink`) *before* the warm-up starts, journals every event
    to disk as it happens, and on the way out closes the recording, cuts it with
    the existing session pipeline and writes the run record.

What the controller watches for while a run is going:

* **A stalled display.** The GUI ticks every frame. If two ticks are further
  apart than ``stall_threshold`` (a blocked GUI thread: a dragged window, a
  long callback), the run is paused *at the last tick* -- the last moment the
  screen was known to be current -- so the schedule never moves through
  phases nobody saw. The segment that was running is marked ``interrupted``.
* **A lagging renderer.** If the newest frame actually shown is older than
  ``render_stall_threshold``, the run is paused and the unseen stretch is
  recorded as a ``presentation_gap``; intervals overlapping it are flagged and
  dropped from training.
* **A lost stream** (recording only). If no modality has delivered for
  ``disconnect_timeout``, any one modality for ``modality_silence_limit``, or
  the logger thread has died, the run is paused with reason
  ``collection_disconnected`` and the silent stretch recorded as a
  ``data_gap``. Resume is refused until data flows again.

A stopped, failed or shut-down run is closed at the best timestamp available,
its open intervals marked with the reason, and its repetitions never counted as
completed unless they finished.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import annotations as notes
from . import targets as target_files
from .animation import pose_for
from .clock import ClockMapping
from .schedule import (CANCELLED, COMPLETED, FAILED, PAUSED_STATE, READY, RUNNING,
                       SEGMENT_INTERRUPTED, TERMINAL, ChoreographyScheduler, build_plan)

PREVIEW = "preview"
RECORDING = "recording"
STARTING = "starting"
FINALIZING = "finalizing"

COLLECTION_MODE = "choreography"
#: How often a recording's journal notes that the run is still alive.
HEARTBEAT_SECONDS = 0.5


class CollectionLink:
    """What a recording run needs from the collection service.

    The GUI's implementation drives :class:`~libemg.data_handler.OnlineDataHandler`
    through the same session-logging code the image-folder collection uses.
    """

    def open(self, header):
        """Start the continuous recording. Returns ``{"collection_session_id", "modalities"}``."""
        raise NotImplementedError

    def counts(self):
        """Samples logged so far, per modality."""
        raise NotImplementedError

    def logger_alive(self):
        raise NotImplementedError

    def save_progress(self, stages):
        """Persist the annotations collected so far."""

    def close(self, stages, metadata):
        """Stop the recording, index and cut it. Returns ``{modality: ModalityClock}``."""
        raise NotImplementedError


@dataclass
class RunResult:
    status: str
    reason: str | None
    mode: str
    run_id: str
    intervals: list = field(default_factory=list)
    record_path: Path | None = None
    error: str | None = None


class _Persister(threading.Thread):
    """Saves annotation progress off the GUI thread, coalescing bursts of requests."""

    def __init__(self, work):
        super().__init__(name="choreography-persist", daemon=True)
        self._work = work
        self._dirty = threading.Event()
        self._stopping = False
        self.start()

    def request(self):
        self._dirty.set()

    def run(self):
        while True:
            self._dirty.wait()
            self._dirty.clear()
            if self._stopping:
                return
            try:
                self._work()
            except Exception as error:  # noqa: BLE001 - progress saving must not kill a run
                print(f"LibEMG -> choreography (could not save progress: {error})")

    def stop(self):
        self._stopping = True
        self._dirty.set()
        self.join(timeout=5)


class ChoreographyController:
    """Runs one configuration, as preview or as a recording, as often as it is reset.

    Parameters
    ----------
    assets: ChoreographyAssets
    config: ChoreographyConfig
        Frozen into a snapshot here; later edits to it do not reach this run.
    mode: str
        ``'preview'`` or ``'recording'``.
    link: CollectionLink
        Required for ``'recording'``.
    output_folder: str or Path
        Where a recording's journal and run record go (beside the session files).
    clock, wall: callable
        The monotonic clock the scheduler runs on and the collection clock
        (``time.time``) the logger stamps with.
    stall_threshold: float or None
        Longest gap between ticks before the display counts as stalled.
    render_stall_threshold: float or None
        Oldest a shown frame may be. None until a renderer reports frames.
    """

    def __init__(self, assets, config, mode=PREVIEW, link=None, output_folder=None,
                 clock=time.monotonic, wall=time.time, stall_threshold=0.35,
                 render_stall_threshold=None, disconnect_timeout=1.0,
                 modality_silence_limit=5.0, stream_check_interval=0.1, metadata=None):
        if mode not in (PREVIEW, RECORDING):
            raise ValueError(f"Unknown run mode {mode!r}.")
        if mode == RECORDING and (link is None or output_folder is None):
            raise ValueError("A recording run needs a collection link and an output folder.")
        self.assets = assets
        self.config = config.snapshot()
        self.plan = build_plan(self.config, assets.gesture_ids)
        self.mode = mode
        self.link = link
        self.output_folder = Path(output_folder) if output_folder is not None else None
        self.clock = clock
        self.wall = wall
        self.stall_threshold = stall_threshold
        self.render_stall_threshold = render_stall_threshold
        self.disconnect_timeout = disconnect_timeout
        self.modality_silence_limit = modality_silence_limit
        self.stream_check_interval = stream_check_interval
        self.metadata = dict(metadata or {})
        self.class_ids = {name: number for number, name in assets.class_map.items()}
        self._lock = threading.RLock()
        self.result = None
        self.last_error = None
        self._new_run()

    # -- lifecycle ------------------------------------------------------------------

    def _new_run(self):
        self.run_id = uuid.uuid4().hex
        self.scheduler = ChoreographyScheduler(self.plan, run_id=self.run_id, sink=self._on_event)
        self.clock_map = ClockMapping(monotonic=self.clock, wall=self.wall)
        self.collection_session_id = None
        self.modalities = []
        self._journal = None
        self._persister = None
        self._phase = READY            # controller-level state; see .state
        self._finalized = False
        self._last_tick = None
        self._last_presented = None
        self._last_stream_check = None
        self._last_seen = {}
        self._last_counts = {}
        self._disconnected = False
        self._last_heartbeat = float("-inf")
        self.result = None
        self.header = {}

    @property
    def state(self):
        """``ready``, ``starting``, ``running``, ``paused``, ``finalizing`` or a terminal state."""
        with self._lock:
            if self._phase in (STARTING, FINALIZING):
                return self._phase
            return self.scheduler.state

    @property
    def active(self):
        """A run is under way (or being wound up): settings and mode must not change."""
        return self.state in (STARTING, RUNNING, PAUSED_STATE, FINALIZING)

    @property
    def needs_finalize(self):
        with self._lock:
            return self.scheduler.state in TERMINAL and not self._finalized and self._phase != FINALIZING

    @property
    def recording(self):
        return self.mode == RECORDING

    def _build_header(self):
        return {
            "run_id": self.run_id,
            "collection_session_id": self.collection_session_id,
            "collection_mode": COLLECTION_MODE,
            "run_mode": self.mode,
            "created": datetime.now().isoformat(),
            "asset": self.assets.describe(),
            "class_map": self.assets.class_map_record(),
            "config": self.plan.config.to_dict(),
            "timing": {k: getattr(self.plan.config, k) for k in
                       ("warmup", "ramp_in", "hold", "ramp_out", "rest", "cycle_pause",
                        "cycle_pause_enabled")},
            "shuffle_seed": self.plan.config.shuffle_seed,
            "order_mode": self.plan.config.order_mode,
            "schedule": self.plan.to_dict(),
            "intensity": {"enabled": bool(self.plan.config.intensity_enabled),
                          "levels": list(self.plan.config.intensity_levels),
                          "order": self.plan.config.intensity_order,
                          "unit": "fraction of maximum effort (target shown, not measured)"},
            "targets": target_files.targets_description(self.plan.config.intensity_enabled),
            "modalities": list(self.modalities),
            **self.metadata,
        }

    def start(self):
        """Begin the run: open the recording first (if recording), then the warm-up.

        Raises whatever opening the recording raised, leaving the run ready to
        try again.
        """
        with self._lock:
            if self.state != READY:
                raise RuntimeError(f"Cannot start: the run is {self.state}.")
            self._phase = STARTING
        try:
            if self.recording:
                self.header = self._build_header()
                opened = self.link.open(self.header) or {}
                self.collection_session_id = opened.get("collection_session_id")
                self.modalities = list(opened.get("modalities", []))
                self._journal = notes.RunJournal(self.output_folder / notes.JOURNAL_NAME)
                self._persister = _Persister(self._save_progress)
        except Exception as error:
            with self._lock:
                self._phase = READY
                self.last_error = str(error)
            raise
        with self._lock:
            self.clock_map.anchor()
            self.header = self._build_header()
            if self._journal is not None:
                self._journal.header(self.header)
            now = self.clock()
            self._phase = RUNNING
            self._last_tick = now
            self._last_presented = now
            if self.recording:
                self._last_counts = dict(self.link.counts())
                self._last_seen = {mod: now for mod in (self._last_counts or
                                                        dict.fromkeys(self.modalities))}
                self._last_stream_check = now
            self.scheduler.start(now, mode=self.mode,
                                 collection_session_id=self.collection_session_id)
            self._phase = self.scheduler.state

    def pause(self, now=None, reason="user"):
        with self._lock:
            now = self.clock() if now is None else now
            paused = self.scheduler.pause(now, reason=reason)
            self._last_tick = now
            return paused

    def resume(self, now=None):
        """Continue from where the run was paused. False if it cannot (yet)."""
        with self._lock:
            now = self.clock() if now is None else now
            if self.scheduler.state != PAUSED_STATE:
                return False
            if self.recording and not self.stream_alive(now):
                self.last_error = "No data is arriving from the device yet."
                return False
            resumed = self.scheduler.resume(now)
            self._disconnected = False
            self._last_tick = now
            self._last_presented = now
            return resumed

    def stop(self, now=None, reason="user_stop", status=CANCELLED):
        """End the run early. Call :meth:`finalize` afterwards (the GUI does so off-thread)."""
        with self._lock:
            now = self.clock() if now is None else now
            if self._phase == STARTING:
                return self.state
            return self.scheduler.stop(now, status=status, reason=reason)

    def fail(self, reason, now=None):
        """End the run as failed, e.g. because the renderer died."""
        with self._lock:
            self.last_error = reason
        return self.stop(now=now, reason=reason, status=FAILED)

    def reset(self):
        """Back to a fresh, ready run of the same configuration. Only once this run is wound up."""
        with self._lock:
            if self.scheduler.state not in (READY, *TERMINAL) or self._phase == FINALIZING:
                raise RuntimeError("Stop the run before resetting it.")
            if self.scheduler.state in TERMINAL and not self._finalized:
                raise RuntimeError("The run has not been saved yet.")
            self._new_run()

    def shutdown(self, now=None):
        """The application is closing: end any run and save what it collected, synchronously."""
        if self.state in (RUNNING, PAUSED_STATE):
            self.stop(now=now, reason="application_shutdown")
        if self.needs_finalize:
            self.finalize()

    # -- per frame ------------------------------------------------------------------

    def tick(self, now=None):
        """Advance the run to ``now`` and look for stalls and a lost stream. Returns the view."""
        with self._lock:
            now = self.clock() if now is None else now
            if self.scheduler.state == RUNNING and self._phase == RUNNING:
                self._check_display(now)
                if self.scheduler.state == RUNNING:
                    self.scheduler.tick(now)
                if self.scheduler.state == RUNNING and self.recording:
                    self._check_stream(now)
            elif self.scheduler.state == PAUSED_STATE and self.recording:
                self._poll_counts(now)
            if (self._journal is not None and self.scheduler.state in (RUNNING, PAUSED_STATE)
                    and now - self._last_heartbeat >= HEARTBEAT_SECONDS):
                # So a run that dies between events can still be closed near
                # the moment it died, not at its last phase change.
                self._last_heartbeat = now
                self._journal.heartbeat({"monotonic": now, "collection_time":
                                         self.clock_map.to_collection(now),
                                         "state": self.scheduler.state})
            self._last_tick = now
            if self.scheduler.state in TERMINAL and self._phase == RUNNING:
                self._phase = self.scheduler.state
            return self.scheduler.view(now)

    def view(self, now=None):
        with self._lock:
            return self.scheduler.view(self.clock() if now is None else now)

    def pose(self, view=None):
        """``(clip, clip_time)`` for the renderer."""
        return pose_for(view if view is not None else self.view())

    def frame_presented(self, request_monotonic):
        """The GUI put on screen a frame posed for ``request_monotonic``."""
        with self._lock:
            if self._last_presented is None or request_monotonic > self._last_presented:
                self._last_presented = request_monotonic

    def _check_display(self, now):
        if (self.stall_threshold is not None and self._last_tick is not None
                and now - self._last_tick > self.stall_threshold):
            # Nothing has been drawn since the last tick, so the schedule stops
            # there: the stretch since is a pause, not a phase nobody saw.
            last = self.scheduler.last_time if self.scheduler.last_time is not None else self._last_tick
            self._gap("display_stall", last, now, reason="ui_stall")
            self.scheduler.pause(last, reason="ui_stall", status=SEGMENT_INTERRUPTED)
            return
        if (self.render_stall_threshold is not None and self._last_presented is not None
                and now - self._last_presented > self.render_stall_threshold):
            self._gap("presentation_gap", self._last_presented, now, reason="render_stall")
            self.scheduler.pause(now, reason="render_stall", status=SEGMENT_INTERRUPTED)

    def _poll_counts(self, now):
        counts = dict(self.link.counts())
        for mod, count in counts.items():
            if count > self._last_counts.get(mod, -1):
                self._last_seen[mod] = now
        self._last_counts = counts

    def stream_alive(self, now=None):
        """Every modality is delivering and the logger is running."""
        if not self.recording:
            return True
        now = self.clock() if now is None else now
        if not self.link.logger_alive():
            return False
        if not self._last_seen:
            return True
        newest = max(self._last_seen.values())
        if now - newest > self.disconnect_timeout:
            return False
        return all(now - seen <= self.modality_silence_limit for seen in self._last_seen.values())

    def _check_stream(self, now):
        if (self._last_stream_check is not None
                and now - self._last_stream_check < self.stream_check_interval):
            return
        self._last_stream_check = now
        self._poll_counts(now)
        if not self._last_seen and self.link.logger_alive():
            return
        newest = max(self._last_seen.values()) if self._last_seen else now
        silent = {mod: seen for mod, seen in self._last_seen.items()
                  if now - seen > self.modality_silence_limit}
        if not self.link.logger_alive():
            since, why = newest, "logger_stopped"
        elif now - newest > self.disconnect_timeout:
            since, why = newest, "no_samples"
        elif silent:
            since, why = min(silent.values()), "modality_silent:" + ",".join(sorted(silent))
        else:
            return
        self._disconnected = True
        self._gap("data_gap", since, now, reason=why)
        self.scheduler.pause(now, reason="collection_disconnected", status=SEGMENT_INTERRUPTED)

    def _gap(self, kind, start, end, reason):
        mapped = self.recording and self.clock_map.anchors
        self.scheduler.record(
            "gap", end, gap_kind=kind, reason=reason, start_monotonic=start, end_monotonic=end,
            start_collection_time=self.clock_map.to_collection(start) if mapped else None,
            end_collection_time=self.clock_map.to_collection(end) if mapped else None)

    @property
    def disconnected(self):
        return self._disconnected

    # -- events and persistence -------------------------------------------------------

    def _on_event(self, event):
        if self.recording and self.clock_map.anchors:
            self.clock_map.anchor()
            event["collection_time"] = self.clock_map.to_collection(event["monotonic"])
        else:
            event["collection_time"] = None
        event["collection_session_id"] = self.collection_session_id
        if self._journal is not None:
            self._journal.event(event)
        if self._persister is not None and event["type"] in ("phase_ended", "resumed"):
            self._persister.request()

    def _repetition_lookup(self):
        return {r.repetition_id: (r.motion_id, r.gesture_repetition_index)
                for r in self.plan.repetitions}

    def _repetition_order(self):
        return {r.repetition_id: r.order_index for r in self.plan.repetitions}

    def intervals(self):
        """The run's intervals as of now, built from its events."""
        with self._lock:
            events = list(self.scheduler.events)
        return notes.build_intervals(events, self.collection_session_id, self.class_ids,
                                     self.plan.config.parent_regions)

    def _save_progress(self):
        stages = notes.stages_from_intervals(self.intervals(), self.class_ids,
                                             self._repetition_lookup())
        self.link.save_progress(stages)

    def finalize(self):
        """Wind up a finished run: close the recording, index it, write the run record.

        Safe to call once per run; the GUI calls it on a worker thread because
        cutting a long recording takes a while.
        """
        with self._lock:
            if self._finalized:
                return self.result
            if self.scheduler.state not in TERMINAL:
                raise RuntimeError("The run is still going; stop it first.")
            self._phase = FINALIZING
            status = self.scheduler.state
            end = self.scheduler.events[-1] if self.scheduler.events else {}
            reason = end.get("reason")
            events = list(self.scheduler.events)
        result = RunResult(status=status, reason=reason, mode=self.mode, run_id=self.run_id)
        try:
            intervals = notes.build_intervals(events, self.collection_session_id, self.class_ids,
                                              self.plan.config.parent_regions)
            result.intervals = intervals
            if self.recording:
                if self._persister is not None:
                    self._persister.stop()
                stages = notes.stages_from_intervals(intervals, self.class_ids,
                                                     self._repetition_lookup())
                clocks = {}
                try:
                    clocks = self.link.close(stages, self._session_metadata(status, reason)) or {}
                except Exception as error:      # noqa: BLE001 - still write what we have
                    result.error = f"Closing the recording failed: {error}"
                primary = notes.assign_sample_indices(intervals, clocks)
                written = {}
                try:
                    written = target_files.write_target_files(
                        self.output_folder, intervals, clocks, self._repetition_order(),
                        intensity=self.plan.config.intensity_enabled)
                except Exception as error:      # noqa: BLE001 - the record still matters more
                    result.error = (result.error or "") + f" Writing target files failed: {error}"
                record = notes.run_record(
                    self.header, events, intervals, status, reason, clocks, primary,
                    extra={"finished": datetime.now().isoformat(),
                           "clock": self.clock_map.to_dict(),
                           "target_files": written,
                           "error": result.error or self.last_error})
                result.record_path = notes.write_run_record(
                    self.output_folder / notes.RUN_RECORD_NAME, record)
                if self._journal is not None:
                    self._journal.footer({"status": status, "reason": reason,
                                          "run_record": notes.RUN_RECORD_NAME})
                    self._journal.close()
        finally:
            with self._lock:
                self._finalized = True
                self._phase = status
                self.result = result
        return result

    def _session_metadata(self, status, reason):
        return {"choreography_run": {"run_id": self.run_id, "status": status, "end_reason": reason,
                                     "record": notes.RUN_RECORD_NAME,
                                     "journal": notes.JOURNAL_NAME,
                                     "schema_version": notes.RUN_SCHEMA_VERSION}}


__all__ = ["ChoreographyController", "CollectionLink", "RunResult", "PREVIEW", "RECORDING",
           "STARTING", "FINALIZING", "COLLECTION_MODE"]
