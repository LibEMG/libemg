"""Hand-marked contraction regions, saved beside the recordings they describe.

A recording made outside a cued protocol -- or one where the subject did not
contract when the cue said to -- has no reliable record of when the
contraction actually happened. Someone has to look at it and mark it. This
module is where those marks live: one small JSON file beside each data file,
``aff_0.txt`` -> ``aff_0.annotations.json``, holding the regions as sample
indices.

Sample indices rather than seconds, because the files themselves carry no
sampling rate. The rate the annotator was looking at is recorded too, but the
regions stay correct whatever rate is assumed later.

An annotation file with no regions is meaningful: it says the recording was
looked at and holds no usable contraction. A recording that has not been looked
at has no annotation file at all.

Examples
---------
>>> index = DatasetIndex.scan("B:/data/Shriner_Participants")
>>> entry = index.entries[0]
>>> annotation = entry.load_annotation()     # None until someone marks it
>>> data = load_emg_text(entry.path)
>>> active = [data[start:stop] for start, stop in annotation.regions]
"""

import datetime
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np

#: Appended to a data file's stem to name its annotation file.
ANNOTATION_SUFFIX = ".annotations.json"
#: The file a dataset root may carry naming the subjects to use.
DATASET_DESCRIPTION = "dataset_description.yaml"


def annotation_path(data_path):
    """Where the annotation for a data file is kept.

    Parameters
    ----------
    data_path: str or Path
        The recording, e.g. ``.../Pinch/aff_0.txt``.

    Returns
    ----------
    Path
        ``.../Pinch/aff_0.annotations.json``.
    """
    data_path = Path(data_path)
    return data_path.with_name(data_path.stem + ANNOTATION_SUFFIX)


def normalize_regions(regions, num_samples=None):
    """Put regions in a form that can be saved and sliced with.

    Each region becomes an integer ``(start, stop)`` with ``start < stop``
    (stop exclusive), clipped to the recording when its length is known.
    Regions that overlap or touch are merged, since two marks over the same
    samples mean one contraction, and empty ones are dropped.

    Parameters
    ----------
    regions: iterable of (start, stop)
        In samples, either order.
    num_samples: int or None (optional)
        The recording length to clip to.

    Returns
    ----------
    list of tuple
        Sorted, non-overlapping ``(start, stop)`` pairs.
    """
    cleaned = []
    for start, stop in regions:
        start, stop = sorted((int(round(start)), int(round(stop))))
        start = max(start, 0)
        if num_samples is not None:
            stop = min(stop, int(num_samples))
        if stop > start:
            cleaned.append((start, stop))
    cleaned.sort()
    merged = []
    for start, stop in cleaned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


@dataclass
class ContractionAnnotation:
    """The contraction regions marked on one recording.

    Parameters
    ----------
    regions: list of (start, stop)
        Sample indices, stop exclusive.
    num_samples: int or None
        The length of the recording when it was annotated, so an annotation
        that no longer matches its file can be noticed.
    sampling_rate: float or None
        The rate the annotator's time axis assumed. Informational only.
    file: str
        The data file's name.
    notes: str
        Anything the annotator wanted to say about the recording.
    updated: str
        When it was last saved (ISO 8601).
    """
    regions: list = field(default_factory=list)
    num_samples: int = None
    sampling_rate: float = None
    file: str = ""
    notes: str = ""
    updated: str = ""

    def to_dict(self):
        return {
            "file": self.file,
            "num_samples": self.num_samples,
            "sampling_rate": self.sampling_rate,
            "regions": [{"start": int(a), "stop": int(b)} for a, b in self.regions],
            "notes": self.notes,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, raw):
        regions = []
        for region in raw.get("regions", []):
            if isinstance(region, dict):
                regions.append((region["start"], region["stop"]))
            else:
                regions.append(tuple(region))
        return cls(regions=normalize_regions(regions, raw.get("num_samples")),
                   num_samples=raw.get("num_samples"),
                   sampling_rate=raw.get("sampling_rate"),
                   file=raw.get("file", ""),
                   notes=raw.get("notes", ""),
                   updated=raw.get("updated", ""))

    def save(self, data_path):
        """Write this annotation beside ``data_path``.

        Written to a temporary file and moved into place, so a crash mid-write
        cannot leave a half-written file that reads as annotated.

        Returns
        ----------
        Path
            The annotation file.
        """
        path = annotation_path(data_path)
        self.file = Path(data_path).name
        self.regions = normalize_regions(self.regions, self.num_samples)
        self.updated = datetime.datetime.now().isoformat(timespec="seconds")
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        os.replace(temporary, path)
        return path

    @classmethod
    def load(cls, data_path):
        """Read the annotation beside ``data_path``, or None if there is none."""
        path = annotation_path(data_path)
        if not path.exists():
            return None
        with open(path) as handle:
            return cls.from_dict(json.load(handle))

    def mask(self, num_samples=None):
        """A boolean array, True inside a contraction region."""
        length = num_samples if num_samples is not None else self.num_samples
        if length is None:
            raise ValueError("The recording length is needed to build a mask.")
        mask = np.zeros(int(length), dtype=bool)
        for start, stop in self.regions:
            mask[start:stop] = True
        return mask


@dataclass
class IndexEntry:
    """One recording in a dataset.

    Parameters
    ----------
    path: Path
        The data file.
    relative: str
        Its path under the dataset root, with forward slashes.
    fields: dict
        What its location says about it -- ``{"subject": ..., "task": ...}``
        -- named by the dataset's path template, or by folder level when it
        has none.
    """
    path: Path
    relative: str
    fields: dict = field(default_factory=dict)

    @property
    def subject(self):
        return self.fields.get("subject", "")

    @property
    def annotation_path(self):
        return annotation_path(self.path)

    @property
    def is_annotated(self):
        return self.annotation_path.exists()

    def load_annotation(self):
        return ContractionAnnotation.load(self.path)

    @property
    def label(self):
        return self.relative.replace("/", " / ")


#: Used when neither the dataset description nor the caller says which files
#: are recordings.
DEFAULT_SPEC = {"data_files": ["**/*.txt"], "exclude_files": [], "path_template": None}


@dataclass
class DatasetIndex:
    """Every recording under a dataset root, as its description defines them.

    Which files are recordings is not built in, because it changes from
    project to project. It comes from the ``annotation`` block of the root's
    ``dataset_description.yaml`` (every key optional)::

        used_subjects:              # top-level folders to include
          - "Participant_03"
        annotation:
          data_files: "*/*/*.txt"   # glob(s) under the root; default **/*.txt
          exclude_files:            # glob(s) to leave out
            - "*/test/*"
          path_template: "{subject}/{task}/{side}_{rep}.txt"

    Globs are matched against the whole path under the root: ``*`` stays
    within one folder and ``**`` spans any number of them. The path template
    names the parts of a recording's path; the names become the index's
    columns and filters, and a file that does not fit it is left out (and
    listed in :attr:`unmatched`, so a stray file is noticed). Without a
    template the first folder is the subject and the rest is the folder.

    Parameters
    ----------
    root: Path
        The dataset folder.
    entries: list of IndexEntry
        In path order.
    field_names: list of str
        The fields entries carry, in template order.
    skipped_subjects: list of str
        Subjects left out because ``used_subjects`` does not list them.
    unmatched: list of str
        Files the globs picked up that do not fit the path template.
    spec: dict
        The rules the index was built with.
    """
    root: Path
    entries: list = field(default_factory=list)
    field_names: list = field(default_factory=list)
    skipped_subjects: list = field(default_factory=list)
    unmatched: list = field(default_factory=list)
    spec: dict = field(default_factory=dict)

    @classmethod
    def scan(cls, root, spec=None, used_only=True):
        """Find the recordings under ``root``.

        Parameters
        ----------
        root: str or Path
            The dataset folder.
        spec: dict or None (optional)
            ``data_files``, ``exclude_files`` and ``path_template`` to use in
            place of the description's -- for a dataset whose description
            cannot be edited. Keys left out still come from the description.
        used_only: bool (optional), default=True
            Honour the description's ``used_subjects`` list.

        Returns
        ----------
        DatasetIndex
        """
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"{root} is not a folder.")
        description = read_dataset_description(root)
        rules = dict(DEFAULT_SPEC)
        rules.update({k: v for k, v in (description.get("annotation") or {}).items()
                      if v is not None})
        rules.update({k: v for k, v in (spec or {}).items() if v is not None})
        includes = _as_list(rules["data_files"])
        excludes = _as_list(rules["exclude_files"])
        template = rules.get("path_template")
        parse = _template_regex(template) if template else None
        used = description.get("used_subjects") if used_only else None
        used = None if used is None else {str(s) for s in used}

        entries, unmatched, skipped = [], [], set()
        for path in root.rglob("*"):
            relative = path.relative_to(root).as_posix()
            pure = PurePosixPath(relative)
            if not any(pure.full_match(g) for g in includes):
                continue
            if any(pure.full_match(g) for g in excludes) or not path.is_file():
                continue
            if parse is not None:
                match = parse.fullmatch(relative)
                if match is None:
                    unmatched.append(relative)
                    continue
                fields = match.groupdict()
            else:
                fields = _folder_fields(pure)
            subject = fields.get("subject")
            if used is not None and subject is not None and subject not in used:
                skipped.add(subject)
                continue
            entries.append(IndexEntry(path=path, relative=relative, fields=fields))

        entries.sort(key=lambda e: _natural_key(e.relative))
        if parse is not None:
            names = list(parse.groupindex)
        else:
            names = [n for n in ("subject", "folder") if any(n in e.fields for e in entries)]
        return cls(root=root, entries=entries, field_names=names,
                   skipped_subjects=sorted(skipped), unmatched=sorted(unmatched),
                   spec=rules)

    def progress(self):
        """``(annotated, total)`` over every entry."""
        return sum(e.is_annotated for e in self.entries), len(self.entries)


def read_dataset_description(root):
    """The root's ``dataset_description.yaml`` as a dict, empty if it has none."""
    description = Path(root) / DATASET_DESCRIPTION
    if not description.exists():
        return {}
    import yaml
    with open(description) as handle:
        return yaml.safe_load(handle) or {}


def _as_list(value):
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _template_regex(template):
    """``"{subject}/{task}/{side}_{rep}.txt"`` as a regex with a group per name.

    A name matches within one folder and as little as it can, so
    ``{side}_{rep}`` splits ``unaff_0`` at the underscore. A name used twice
    must match the same text both times.
    """
    pattern, seen, position = "", set(), 0
    for match in re.finditer(r"\{(\w+)\}", template):
        pattern += re.escape(template[position:match.start()])
        name = match.group(1)
        pattern += f"(?P={name})" if name in seen else f"(?P<{name}>[^/]+?)"
        seen.add(name)
        position = match.end()
    pattern += re.escape(template[position:])
    return re.compile(pattern)


def _folder_fields(pure):
    """Fields for a file with no template: first folder is the subject."""
    folders = pure.parts[:-1]
    fields = {}
    if folders:
        fields["subject"] = folders[0]
    if len(folders) > 1:
        fields["folder"] = "/".join(folders[1:])
    return fields


def _natural_key(text):
    """Sort ``rep_2`` before ``rep_10``."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", text)]


def load_emg_text(path, delimiter=","):
    """Read a delimited text recording into a ``(samples, channels)`` array.

    pandas parses a recording of a few hundred thousand rows several times
    faster than numpy, so it is used when it is installed.
    """
    try:
        import pandas as pd
    except ImportError:
        data = np.loadtxt(path, delimiter=delimiter, ndmin=2)
    else:
        data = pd.read_csv(path, delimiter=delimiter, header=None).to_numpy(dtype=float)
    return np.atleast_2d(data)
