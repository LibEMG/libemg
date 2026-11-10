"""Coverage for the annotation file that says what a continuous recording holds."""
import json
from pathlib import Path

import numpy as np
import pytest

from libemg.session import (ANNOTATIONS_NAME, PAUSE, PROMPT, REST,
                            ModalityClock, SessionAnnotations, Stage,
                            cut_session, fit_production_clock, main)


def _recording(path, fs=1000.0, seconds=2.0, batch=25, channels=2, t0=1000.0,
               lag=0.004, jitter=0.003, seed=0):
    """Write a session file for a link delivering in batches, each one late.

    The first channel is the sample's own index, so a cut file says where in the
    recording it came from.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds) // batch * batch
    index = np.arange(n, dtype=float)
    group = index // batch
    delay = lag + jitter * rng.random(int(group[-1]) + 1)[group.astype(int)]
    arrival = t0 + ((group + 1) * batch - 1) / fs + delay
    samples = np.column_stack([index + 1000 * c for c in range(channels)])
    np.savetxt(path, np.column_stack([arrival, samples]), delimiter=",")
    return t0, fs, n


def _session(tmp_path, seconds=2.0, **kwargs):
    t0, fs, n = _recording(tmp_path / "_session_emg.csv", seconds=seconds, **kwargs)
    session = SessionAnnotations(folder=tmp_path, modalities=["emg"],
                                 class_map={"0": "flexion", "1": "extension"})
    # Two prompts, each with a rest in front of it, then a pause.
    session.stages = [
        Stage(kind=REST, label="flexion", motion=0, rep=0,
              t_start=t0 + 0.05, t_end=t0 + 0.3, duration=0.25),
        Stage(kind=PROMPT, label="flexion", motion=0, rep=0,
              t_start=t0 + 0.3, t_end=t0 + 0.81, duration=0.5),
        Stage(kind=REST, label="extension", motion=1, rep=0,
              t_start=t0 + 0.81, t_end=t0 + 1.06, duration=0.25),
        Stage(kind=PROMPT, label="extension", motion=1, rep=0,
              t_start=t0 + 1.06, t_end=t0 + 1.57, duration=0.5),
        Stage(kind=PAUSE, label="", motion=None, rep=0,
              t_start=t0 + 1.57, t_end=t0 + 1.8, duration=0.23),
    ]
    session.resolve()
    return session, t0, fs, n


def test_the_clock_fit_sees_past_the_delivery_lag(tmp_path):
    t0, fs, n = _recording(tmp_path / "_session_emg.csv", seconds=10.0, lag=0.02,
                           jitter=0.05)
    arrivals = np.loadtxt(tmp_path / "_session_emg.csv", delimiter=",")[:, 0]

    measured_fs, anchor, spread = fit_production_clock(arrivals)

    assert measured_fs == pytest.approx(fs, rel=2e-3)
    # The anchor sits at the origin of production, not at the mean arrival, so a
    # cut taken from it is not systematically late by the average lag.
    assert anchor == pytest.approx(t0, abs=0.03)
    assert spread == pytest.approx(0.05, abs=0.02)


def test_the_clock_fit_declines_a_recording_it_cannot_measure():
    assert fit_production_clock(np.array([])) is None
    assert fit_production_clock(np.array([1.0])) is None
    # One batch is not a rate, however many samples are in it.
    assert fit_production_clock(np.full(500, 12.0)) is None


def test_indices_follow_the_stages_and_are_clamped_to_the_recording(tmp_path):
    session, t0, fs, n = _session(tmp_path)

    prompt = session.select(kinds=(PROMPT,))[0]
    start, end = prompt.span("emg")
    assert start == pytest.approx(0.3 * fs, abs=fs * 0.01)
    # Indexed to the duration it was asked for, so both takes of the same prompt
    # are the same length whatever the render loop did.
    assert end - start == round(0.5 * fs)
    assert prompt.indices["emg"]["end_observed"] > end

    overrunning = Stage(kind=PROMPT, label="x", motion=9, rep=0,
                        t_start=t0 + 1.9, t_end=t0 + 4.0, duration=2.0)
    session.stages.append(overrunning)
    session.assign_indices()
    assert overrunning.span("emg")[1] == n, "a stage past the end is clipped to it"


def test_a_saved_session_reloads_without_re_reading_the_recording(tmp_path):
    session, t0, fs, n = _session(tmp_path)
    session.save()

    reloaded = SessionAnnotations.load(tmp_path / ANNOTATIONS_NAME)

    assert reloaded.class_map == session.class_map
    assert [s.kind for s in reloaded.stages] == [s.kind for s in session.stages]
    assert reloaded.clocks["emg"].fs == pytest.approx(session.clocks["emg"].fs)
    # Already measured, so nothing needs re-reading and nothing is missing.
    assert reloaded.measure_clocks() == []
    # And the folder comes from where the file was found, so a session that has
    # been moved or copied still cuts.
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for name in ("_session_emg.csv", ANNOTATIONS_NAME):
        (moved / name).write_bytes((tmp_path / name).read_bytes())
    assert SessionAnnotations.load(moved).source_path("emg") == moved / "_session_emg.csv"


def test_loading_refuses_a_file_from_a_newer_libemg(tmp_path):
    path = tmp_path / ANNOTATIONS_NAME
    path.write_text(json.dumps({"schema": 99, "stages": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="newer LibEMG"):
        SessionAnnotations.load(path)


def test_a_session_interrupted_before_it_was_indexed_can_still_be_cut(tmp_path):
    session, t0, fs, n = _session(tmp_path)
    # What a collection that died mid-stage leaves: stages with wall-clock times
    # and no measured clock, because nothing got as far as fitting one.
    for stage in session.stages:
        stage.indices = {}
    session.clocks = {}
    session.save()

    written = cut_session(tmp_path, output_folder=tmp_path / "recut")

    assert len(written) == 2
    for cut in written:
        chunk = np.loadtxt(cut.path, delimiter=",", ndmin=2)
        assert len(chunk) == cut.wanted
        assert chunk[0, 0] == pytest.approx(cut.stage.span("emg")[0], abs=1)


def test_cut_slices_the_recording_at_the_annotated_indices(tmp_path):
    session, t0, fs, n = _session(tmp_path)
    _, samples = session.read("emg")

    written = session.cut(output_folder=tmp_path / "recut")

    assert {cut.path.name for cut in written} == {"C_0_R_0_emg.csv", "C_1_R_0_emg.csv"}
    for cut in written:
        start, end = cut.stage.span("emg")
        np.testing.assert_array_equal(np.loadtxt(cut.path, delimiter=",", ndmin=2),
                                      samples[start:end])


def test_cut_can_write_the_time_each_sample_was_produced(tmp_path):
    session, t0, fs, n = _session(tmp_path)

    cut = session.cut(output_folder=tmp_path / "stamped", timestamps=True)[0]
    chunk = np.loadtxt(cut.path, delimiter=",", ndmin=2)

    start = cut.stage.span("emg")[0]
    clock = session.clocks["emg"]
    # The produced clock, evenly spaced -- not the arrival time of a whole batch
    # repeated down the column, which is what the session file carries.
    assert chunk[0, 0] == pytest.approx(clock.anchor + start / clock.fs)
    np.testing.assert_allclose(np.diff(chunk[:, 0]), 1 / clock.fs, rtol=1e-9)
    np.testing.assert_array_equal(chunk[:, 1], np.arange(start, start + len(chunk)))


def test_the_timeline_labels_every_sample_exactly_once(tmp_path):
    session, t0, fs, n = _session(tmp_path)

    spans = session.timeline("emg")

    assert spans[0].start == 0 and spans[-1].end == n
    for before, after in zip(spans, spans[1:]):
        assert before.end == after.start
    # The recording starts before the first stage does, and that stretch is
    # accounted for rather than left out.
    assert spans[0].kind == PAUSE and len(spans) == len(session.stages) + 1
    assert sum(len(span) for span in spans) == n


def test_superseded_takes_are_kept_but_not_cut(tmp_path):
    session, t0, fs, n = _session(tmp_path)
    first = session.select(kinds=(PROMPT,))[0]
    first.superseded = True

    assert len(session.cut(output_folder=tmp_path / "recut")) == 1
    assert len(session.cut(output_folder=tmp_path / "all", include_superseded=True)) == 2
    # Still in the timeline either way: the samples are in the recording whether
    # or not the take was abandoned.
    assert sum(1 for span in session.timeline("emg") if span.kind == PROMPT) == 2


def test_a_modality_with_no_recording_is_reported_rather_than_cut(tmp_path):
    session, t0, fs, n = _session(tmp_path)
    session.modalities.append("imu")

    assert session.measure_clocks() == ["imu"]
    assert all(cut.modality == "emg" for cut in session.cut(output_folder=tmp_path / "x"))


def test_the_command_line_shows_and_cuts_a_session(tmp_path, capsys):
    session, t0, fs, n = _session(tmp_path)
    session.save()

    assert main(["show", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert "2 prompt" in shown and "flexion" in shown

    assert main(["cut", str(tmp_path / ANNOTATIONS_NAME),
                 "--output-folder", str(tmp_path / "recut"),
                 "--kinds", PROMPT, REST, "--lead-in", "0.1"]) == 0
    assert "Wrote 4 files" in capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "recut").glob("*.csv")) == [
        "C_0_R_0_emg.csv", "C_1_R_0_emg.csv", "rest_0_0_emg.csv", "rest_1_0_emg.csv"]


def test_index_at_clamps_to_the_samples_that_exist():
    clock = ModalityClock(fs=100.0, anchor=50.0, n_samples=1000)
    assert clock.index_at(50.0) == 0
    assert clock.index_at(40.0) == 0
    assert clock.index_at(55.0) == 500
    assert clock.index_at(500.0) == 1000
