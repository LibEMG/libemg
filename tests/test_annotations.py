"""Coverage for hand-marked contraction regions and the panel that marks them."""
import json
import time

import numpy as np
import pytest

from libemg.annotations import (ContractionAnnotation, DatasetIndex,
                                annotation_path, load_emg_text,
                                normalize_regions)
from libemg._gui._annotation_panel import decimate_for_display, moving_rms


SHRINERS_BLOCK = """annotation:
  data_files: "*/*/*.txt"
  path_template: "{subject}/{task}/{side}_{rep}.txt"
"""


def _dataset(root, used=("S1",), block=SHRINERS_BLOCK):
    """Two subjects, one listed as used, with aff/unaff files per task.

    Each subject folder also holds a notes .txt that is not a recording, which
    the description's ``annotation`` block has to keep out.
    """
    rng = np.random.default_rng(0)
    for subject in ("S1", "S2"):
        for task in ("Pinch", "Key"):
            folder = root / subject / task
            folder.mkdir(parents=True)
            for name in ("aff_0.txt", "aff_1.txt", "unaff_0.txt", "unaff_1.txt"):
                np.savetxt(folder / name, rng.normal(size=(500, 3)), delimiter=",")
            (folder / f"{task}01.mat").write_bytes(b"")
        (root / subject / "notes.txt").write_text("not a recording")
        (root / subject / "test").mkdir()
    (root / "dataset_description.yaml").write_text(
        "used_subjects:\n" + "".join(f'  - "{s}"\n' for s in used) + block)
    return root


# --- regions -------------------------------------------------------------

def test_regions_are_ordered_clipped_and_merged():
    regions = normalize_regions([(900, 1200), (50, 10), (100, 150), (140, 300), (5, 5)],
                                num_samples=1000)
    assert regions == [(10, 50), (100, 300), (900, 1000)]


def test_an_annotation_round_trips_beside_its_file(tmp_path):
    data_file = tmp_path / "aff_0.txt"
    data_file.write_text("0\n")
    ContractionAnnotation(regions=[(30, 10), (50, 80)], num_samples=100,
                          sampling_rate=2000, notes="late onset").save(data_file)

    assert annotation_path(data_file) == tmp_path / "aff_0.annotations.json"
    raw = json.loads(annotation_path(data_file).read_text())
    assert raw["file"] == "aff_0.txt"
    assert raw["regions"] == [{"start": 10, "stop": 30}, {"start": 50, "stop": 80}]

    loaded = ContractionAnnotation.load(data_file)
    assert loaded.regions == [(10, 30), (50, 80)]
    assert loaded.notes == "late onset" and loaded.num_samples == 100
    assert loaded.mask().sum() == 50


def test_no_file_means_not_annotated_but_an_empty_file_means_annotated(tmp_path):
    data_file = tmp_path / "aff_0.txt"
    data_file.write_text("0\n")
    assert ContractionAnnotation.load(data_file) is None
    ContractionAnnotation(regions=[], num_samples=1).save(data_file)
    assert ContractionAnnotation.load(data_file).regions == []


# --- index ---------------------------------------------------------------

def test_the_description_decides_which_files_are_recordings(tmp_path):
    index = DatasetIndex.scan(_dataset(tmp_path))
    assert index.skipped_subjects == ["S2"] and index.unmatched == []
    assert len(index.entries) == 8
    assert index.field_names == ["subject", "task", "side", "rep"]
    first = index.entries[0]
    assert first.relative == "S1/Key/aff_0.txt"
    assert first.fields == {"subject": "S1", "task": "Key", "side": "aff", "rep": "0"}

    everything = DatasetIndex.scan(tmp_path, used_only=False)
    assert len(everything.entries) == 16 and everything.skipped_subjects == []


def test_without_a_block_every_txt_is_a_recording_named_by_folder(tmp_path):
    index = DatasetIndex.scan(_dataset(tmp_path, block=""))
    # The notes file is picked up too: nothing said it is not a recording.
    assert len(index.entries) == 9
    assert index.field_names == ["subject", "folder"]
    assert {e.relative for e in index.entries if "folder" not in e.fields} == {"S1/notes.txt"}


def test_files_that_do_not_fit_the_template_are_reported_not_hidden(tmp_path):
    root = _dataset(tmp_path)
    # A template field matches any text in one folder, so only a name of
    # another shape is stray: R_1.txt would fit {side}_{rep}.txt.
    (root / "S1" / "Key" / "readme.txt").write_text("0\n")
    index = DatasetIndex.scan(root)
    assert index.unmatched == ["S1/Key/readme.txt"]
    assert all(e.path.name != "readme.txt" for e in index.entries)


def test_a_layout_change_needs_only_a_new_description(tmp_path):
    """The same root re-laid out as one file per rep, as the Shriners data now is."""
    for subject in ("S1", "S2"):
        for task in ("pinch", "key"):
            folder = tmp_path / subject / task
            folder.mkdir(parents=True)
            for rep in (1, 2, 10):
                (folder / f"R_{rep}.txt").write_text("0\n")
    (tmp_path / "dataset_description.yaml").write_text(
        'used_subjects: ["S2"]\n'
        "annotation:\n"
        '  path_template: "{subject}/{task}/R_{rep}.txt"\n'
        '  exclude_files: ["*/key/*"]\n')
    index = DatasetIndex.scan(tmp_path)
    assert [e.relative for e in index.entries] == [
        "S2/pinch/R_1.txt", "S2/pinch/R_2.txt", "S2/pinch/R_10.txt"]
    assert index.field_names == ["subject", "task", "rep"]


def test_a_spec_from_the_caller_overrides_the_description(tmp_path):
    root = _dataset(tmp_path)
    index = DatasetIndex.scan(root, spec={"exclude_files": "**/unaff_*.txt"})
    assert len(index.entries) == 4
    assert {e.fields["side"] for e in index.entries} == {"aff"}
    # Keys the caller leaves out still come from the description.
    assert index.field_names == ["subject", "task", "side", "rep"]


def test_the_subject_can_sit_anywhere_in_the_path(tmp_path):
    for subject in ("A", "B"):
        folder = tmp_path / "raw" / "pinch" / subject
        folder.mkdir(parents=True)
        (folder / "trial.txt").write_text("0\n")
    (tmp_path / "dataset_description.yaml").write_text(
        'used_subjects: ["B"]\n'
        'annotation:\n  path_template: "raw/{task}/{subject}/trial.txt"\n')
    index = DatasetIndex.scan(tmp_path)
    assert [e.subject for e in index.entries] == ["B"]
    assert index.skipped_subjects == ["A"]


def test_progress_counts_annotation_files(tmp_path):
    index = DatasetIndex.scan(_dataset(tmp_path))
    assert index.progress() == (0, 8)
    ContractionAnnotation(regions=[(0, 10)], num_samples=500).save(index.entries[3].path)
    assert index.progress() == (1, 8)
    assert index.entries[3].is_annotated and not index.entries[0].is_annotated


def test_text_recordings_load_as_samples_by_channels(tmp_path):
    path = tmp_path / "aff_0.txt"
    np.savetxt(path, np.arange(12.0).reshape(4, 3), delimiter=",")
    assert load_emg_text(path).shape == (4, 3)


# --- display -------------------------------------------------------------

def test_decimation_keeps_a_spike_narrower_than_a_bin():
    signal = np.zeros((100_000, 2))
    signal[54_321, 1] = 7.0
    x, y = decimate_for_display(signal, 0, len(signal), 1000)
    assert len(x) <= 1000 and y.shape[1] == 2
    assert y[:, 1].max() == 7.0 and y[:, 0].max() == 0.0
    assert x[0] == 0 and np.all(np.diff(x) >= 0)


def test_short_ranges_are_drawn_sample_for_sample():
    signal = np.arange(20.0).reshape(10, 2)
    x, y = decimate_for_display(signal, 2, 6, 1000)
    assert x.tolist() == [2, 3, 4, 5] and np.array_equal(y, signal[2:6])


def test_the_envelope_rises_over_a_burst():
    signal = np.zeros((4000, 1))
    signal[2000:3000, 0] = np.sin(np.arange(1000))
    envelope = moving_rms(signal, 100)
    assert envelope.shape == signal.shape
    assert envelope[2500, 0] > 0.5 > envelope[500, 0]


# --- the panel, in a real (headless) DearPyGui context -------------------

@pytest.fixture
def context():
    dpg = pytest.importorskip("dearpygui.dearpygui")
    dpg.create_context()
    yield dpg
    dpg.destroy_context()


def _wait_loaded(panel, timeout=10.0):
    deadline = time.time() + timeout
    while panel.data is None and time.time() < deadline:
        panel.poll()
        time.sleep(0.01)
    assert panel.data is not None, "the recording never loaded"


def test_marking_saving_and_reopening_a_recording(context, tmp_path):
    from libemg._gui._annotation_panel import PROGRESS, AnnotationPanel

    dpg = context
    root = _dataset(tmp_path)
    panel = AnnotationPanel(dataset_folder=str(root), sampling_rate=100.0)
    panel.spawn_window()
    assert len(panel.visible) == 8
    assert dpg.get_value(PROGRESS) == 0.0
    # A filter per field with more than one value; subject has only S1.
    assert panel._filter_fields == ["task", "side", "rep"]
    dpg.set_value("__annot_filter_side", "unaff")
    panel._rebuild_table()
    assert len(panel.visible) == 4 and all(e.fields["side"] == "unaff" for e in panel.visible)
    dpg.set_value("__annot_filter_side", "All")
    panel._rebuild_table()

    entry = panel.visible[0]
    panel.open(entry)
    _wait_loaded(panel)
    assert panel.regions == [] and not panel.dirty

    panel.add_region(120, 40)
    panel.add_region(300, 350)
    assert panel.dirty
    panel.save()
    assert not panel.dirty
    assert ContractionAnnotation.load(entry.path).regions == [(40, 120), (300, 350)]
    assert dpg.get_value(PROGRESS) == pytest.approx(1 / 8)

    # Opening another file and coming back shows what was saved.
    panel.open(panel.visible[1])
    _wait_loaded(panel)
    assert panel.regions == []
    panel.open(entry)
    _wait_loaded(panel)
    assert panel.regions == [[40, 120], [300, 350]]

    # A fresh panel -- a later session -- sees the same.
    panel.cleanup()
    later = AnnotationPanel(dataset_folder=str(root)).spawn_window()
    assert dpg.get_value(PROGRESS) == pytest.approx(1 / 8)
    later.open(entry)
    _wait_loaded(later)
    assert later.regions == [[40, 120], [300, 350]]


def test_unsaved_edits_are_not_dropped_by_opening_another_file(context, tmp_path):
    from libemg._gui._annotation_panel import CONFIRM_WINDOW, AnnotationPanel

    dpg = context
    panel = AnnotationPanel(dataset_folder=str(_dataset(tmp_path))).spawn_window()
    first, second = panel.visible[:2]
    panel.open(first)
    _wait_loaded(panel)
    panel.add_region(10, 20)

    panel.open(second)
    assert dpg.does_item_exist(CONFIRM_WINDOW)
    assert panel.current is first and panel.regions == [[10, 20]]


def test_save_and_next_moves_to_an_unannotated_file(context, tmp_path):
    from libemg._gui._annotation_panel import AnnotationPanel

    panel = AnnotationPanel(dataset_folder=str(_dataset(tmp_path))).spawn_window()
    ContractionAnnotation(regions=[], num_samples=500).save(panel.visible[1].path)
    panel.open(panel.visible[0])
    _wait_loaded(panel)
    panel.save(then_next=True)
    # visible[1] is already annotated, so the next one opened is visible[2].
    assert panel.current is panel.visible[2]
