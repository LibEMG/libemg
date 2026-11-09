"""The right-hand choreography asset: its model, its manifest and its class IDs.

The asset is three files shipped under ``libemg/resources/choreography``:

``right_hand.glb``
    A skinned hand (17 joints) with one animation clip per motion. Every motion
    clip follows the same 2.5 s contract -- rest until 0.3 s, the target pose at
    1.0 s, held until 1.5 s, back to rest by 2.2 s, rest to 2.5 s.
``hand_manifest.json``
    The motion IDs, their readable labels and that timing contract. The pose
    quaternions it also carries are Blender bone-local values and are *not*
    glTF joint rotations, so nothing here uses them for playback.
``motion_class_map.json``
    The versioned numeric class ID of every motion. IDs come from this file,
    never from which gestures were ticked or the order they were listed in, so
    ``C_5_R_0_emg.csv`` means wrist flexion in every session collected under the
    same map version.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from libemg.resources import read_resource_bytes, resource_path

RESOURCE_FOLDER = "choreography"
GLB_NAME = "right_hand.glb"
MANIFEST_NAME = "hand_manifest.json"
CLASS_MAP_NAME = "motion_class_map.json"

#: The relaxed hand. Shown during warm-up, rests and cycle pauses; never a
#: selectable motion.
REST_ID = "rest"
#: Clips in the GLB that are not motions a participant can be asked to make.
EXCLUDED_CLIPS = (REST_ID, "Pose_Library_Preview")

# The clip contract every motion clip was exported under (seconds into the
# clip). The scheduler's phases are mapped onto these intervals rather than
# playing the clip at a different speed -- see libemg.choreography.animation.
CLIP_REST = 0.0
CLIP_RAMP_IN_START = 0.3
CLIP_TARGET = 1.0
CLIP_HOLD_END = 1.5
CLIP_RETURNED = 2.2
CLIP_DURATION = 2.5


class AssetError(ValueError):
    """The choreography asset is missing, unreadable or breaks the clip contract."""


@dataclass(frozen=True)
class Gesture:
    """One selectable motion: its stable ID, what the UI calls it, and its class ID."""

    id: str
    label: str
    class_id: int


@dataclass
class ChoreographyAssets:
    """Everything the choreography mode reads from the packaged asset.

    Attributes
    ----------
    glb_path: Path
        The model, inside the installed package.
    manifest: dict
        ``hand_manifest.json`` as loaded.
    gestures: list of Gesture
        The selectable motions, in manifest order. ``rest`` and the preview
        timeline are not among them.
    class_map: dict
        Numeric class ID (int) to stable motion ID, ``rest`` included.
    class_map_version: str
        Which mapping ``class_map`` is. Written into every session.
    """

    glb_path: Path
    manifest_path: Path
    manifest: dict
    gestures: list
    class_map: dict
    class_map_version: str
    glb_sha256: str
    manifest_sha256: str
    labels: dict = field(default_factory=dict)

    @property
    def gesture_ids(self):
        return [g.id for g in self.gestures]

    def gesture(self, motion_id):
        for g in self.gestures:
            if g.id == motion_id:
                return g
        raise KeyError(motion_id)

    def label(self, motion_id):
        """The readable name of a motion, ``Rest`` for the relaxed hand."""
        return self.labels.get(motion_id, motion_id.replace("_", " ").title())

    def class_id(self, motion_id):
        for number, name in self.class_map.items():
            if name == motion_id:
                return number
        raise KeyError(motion_id)

    def describe(self):
        """What a session records about the asset it was collected with."""
        return {
            "glb": GLB_NAME,
            "glb_sha256": self.glb_sha256,
            "manifest": MANIFEST_NAME,
            "manifest_sha256": self.manifest_sha256,
            "fps": self.manifest.get("fps"),
            "clip_contract": {
                "rest": CLIP_REST, "ramp_in_start": CLIP_RAMP_IN_START,
                "target": CLIP_TARGET, "hold_end": CLIP_HOLD_END,
                "returned": CLIP_RETURNED, "duration": CLIP_DURATION,
            },
            "class_map_version": self.class_map_version,
        }

    def class_map_record(self):
        """The versioned numeric-ID mapping, in the form a session stores it."""
        return {"version": self.class_map_version,
                "classes": {str(k): v for k, v in sorted(self.class_map.items())},
                "labels": {v: self.label(v) for v in self.class_map.values()}}


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_class_map(raw=None):
    """The versioned motion-ID map. ``raw`` overrides the packaged file (tests)."""
    if raw is None:
        raw = json.loads(read_resource_bytes(RESOURCE_FOLDER, CLASS_MAP_NAME))
    classes = {int(k): str(v) for k, v in raw["classes"].items()}
    if len(set(classes.values())) != len(classes):
        raise AssetError("motion_class_map.json maps two class IDs to the same motion.")
    return str(raw["version"]), classes


def _check_clip_contract(clip):
    expected = {"duration": CLIP_DURATION, "target_seconds": CLIP_TARGET,
                "hold_end_seconds": CLIP_HOLD_END, "rest_end_seconds": CLIP_RETURNED}
    for key, value in expected.items():
        if key in clip and abs(float(clip[key]) - value) > 1e-6:
            raise AssetError(
                f"Clip {clip.get('id')!r} has {key}={clip[key]}, but the choreography "
                f"timing is built for {value}. Re-export the asset with the standard "
                f"clip contract.")


def load_assets(manifest_path=None, glb_path=None, class_map=None):
    """Load the packaged choreography asset.

    Parameters
    ----------
    manifest_path, glb_path: str or Path, optional
        Use these files instead of the packaged ones. For tests and for trying a
        re-exported asset; a normal run never passes them.
    class_map: dict, optional
        A class map in the ``motion_class_map.json`` layout, instead of the
        packaged one.

    Returns
    ----------
    ChoreographyAssets
    """
    manifest_path = Path(manifest_path) if manifest_path else resource_path(RESOURCE_FOLDER, MANIFEST_NAME)
    glb_path = Path(glb_path) if glb_path else resource_path(RESOURCE_FOLDER, GLB_NAME)
    for path in (manifest_path, glb_path):
        if not path.is_file():
            raise AssetError(f"The choreography asset is missing {path.name} (looked in {path.parent}).")
    manifest_bytes = manifest_path.read_bytes()
    glb_bytes = glb_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    version, classes = load_class_map(class_map)

    labels = {}
    gestures = []
    known = {name: number for number, name in classes.items()}
    extra = []
    for clip in manifest.get("clips", []):
        motion_id = str(clip["id"])
        labels[motion_id] = str(clip.get("label") or motion_id)
        _check_clip_contract(clip)
        if motion_id in EXCLUDED_CLIPS:
            continue
        if motion_id not in known:
            extra.append(motion_id)
            continue
        gestures.append(Gesture(id=motion_id, label=labels[motion_id],
                                class_id=known[motion_id]))
    labels.setdefault(REST_ID, "Rest")
    if extra:
        # A newer asset with motions this map does not know. They get IDs past
        # every known one, in sorted (not manifest or checkbox) order, and the
        # version says the map was extended so a session collected this way is
        # not mistaken for one under the plain map.
        next_id = max(classes) + 1
        for motion_id in sorted(extra):
            classes[next_id] = motion_id
            gestures.append(Gesture(id=motion_id, label=labels[motion_id], class_id=next_id))
            next_id += 1
        version = f"{version}+ext.{_sha256(','.join(sorted(extra)).encode())[:8]}"
    if REST_ID not in classes.values():
        raise AssetError("The class map has no ID for the rest pose.")
    return ChoreographyAssets(
        glb_path=glb_path, manifest_path=manifest_path, manifest=manifest,
        gestures=gestures, class_map=classes, class_map_version=version,
        glb_sha256=_sha256(glb_bytes), manifest_sha256=_sha256(manifest_bytes),
        labels=labels)


__all__ = ["ChoreographyAssets", "Gesture", "AssetError", "load_assets", "load_class_map",
           "REST_ID", "EXCLUDED_CLIPS", "CLIP_REST", "CLIP_RAMP_IN_START", "CLIP_TARGET",
           "CLIP_HOLD_END", "CLIP_RETURNED", "CLIP_DURATION", "GLB_NAME", "MANIFEST_NAME"]
