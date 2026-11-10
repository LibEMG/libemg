"""A small glTF 2.0 (GLB) reader with skeletal animation and linear-blend skinning.

Only what a skinned, animated character needs: the node hierarchy with its
exported transforms, triangle meshes with their skin weights, the skins'
inverse bind matrices, metallic-roughness material factors, and animation
channels (STEP, LINEAR and CUBICSPLINE; rotations slerped on the shortest arc).

Two rules from the glTF specification matter here and are easy to get wrong:

* A node's exported translation/rotation/scale is its rest transform, and an
  animation channel *replaces* one of those three for one node -- it is not
  applied on top. Joints a clip does not animate keep their exported
  transform, unchanged.
* A skinned mesh is placed by its joints alone. The transform of the node the
  mesh hangs from is ignored, and each joint's skinning matrix is
  ``world(joint) @ inverseBind(joint)``.

Nothing here touches a GPU or a window, so it can be used from any thread and
tested without a display.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_GLB_MAGIC = 0x46546C67
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942

_COMPONENT_TYPES = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
                    5125: np.uint32, 5126: np.float32}
_COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
                     "MAT2": 4, "MAT3": 9, "MAT4": 16}
_TRIANGLES = 4


class GltfError(ValueError):
    """The file is not a glTF this reader can use."""


@dataclass
class Material:
    name: str
    base_color: np.ndarray          # linear RGBA
    metallic: float = 1.0
    roughness: float = 1.0
    double_sided: bool = False


@dataclass
class Primitive:
    """One triangle list, ready to skin."""

    positions: np.ndarray           # (N, 3) float32, bind pose
    normals: np.ndarray             # (N, 3) float32
    indices: np.ndarray             # (M, 3) int32
    material: int
    node: int
    skin: int | None = None
    joints: np.ndarray | None = None    # (N, 4) int, indices into the skin's joint list
    weights: np.ndarray | None = None   # (N, 4) float32, rows summing to 1


@dataclass
class Skin:
    name: str
    joints: np.ndarray              # (J,) node indices
    inverse_bind: np.ndarray        # (J, 4, 4)


@dataclass
class Channel:
    node: int
    path: str                       # translation | rotation | scale
    interpolation: str
    times: np.ndarray               # (K,)
    values: np.ndarray              # (K, C), or (K, 3, C) for CUBICSPLINE


@dataclass
class Clip:
    name: str
    channels: list = field(default_factory=list)

    @property
    def duration(self):
        return max((float(c.times[-1]) for c in self.channels if len(c.times)), default=0.0)


# -- quaternion helpers (x, y, z, w) -----------------------------------------

def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(norm == 0, 1.0, norm)


def quat_slerp(q0, q1, alpha):
    """Spherical interpolation on the shortest arc."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + alpha * (q1 - q0))
    theta = np.arccos(min(1.0, dot))
    sin_theta = np.sin(theta)
    return (np.sin((1 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / sin_theta


def quat_to_matrix(q):
    """Rotation matrices for one quaternion (4,) or many (n, 4)."""
    q = quat_normalize(q)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def trs_matrices(translation, rotation, scale):
    """Local matrices ``T @ R @ S`` for (n, 3), (n, 4), (n, 3) arrays."""
    n = len(translation)
    m = np.zeros((n, 4, 4))
    m[:, :3, :3] = quat_to_matrix(rotation) * np.asarray(scale)[:, None, :]
    m[:, :3, 3] = translation
    m[:, 3, 3] = 1.0
    return m


# -- the model ----------------------------------------------------------------

class GltfModel:
    """A loaded GLB: hierarchy, meshes, skins, materials and clips.

    Build one with :meth:`load`. Evaluate a pose with :meth:`pose`, which
    returns the skinning matrices :meth:`skin_vertices` applies.
    """

    def __init__(self, document, binary):
        self._doc = document
        self._bin = binary
        nodes = document.get("nodes", [])
        self.node_names = [n.get("name", f"node{i}") for i, n in enumerate(nodes)]
        self.parents = np.full(len(nodes), -1, dtype=int)
        for index, node in enumerate(nodes):
            for child in node.get("children", []):
                self.parents[child] = index
        self.rest_translation = np.zeros((len(nodes), 3))
        self.rest_rotation = np.tile([0.0, 0.0, 0.0, 1.0], (len(nodes), 1))
        self.rest_scale = np.ones((len(nodes), 3))
        self._fixed_matrix = {}
        for index, node in enumerate(nodes):
            if "matrix" in node:
                # A node given as a matrix cannot be animated (glTF 2.0, 5.25),
                # so it is kept as the matrix it was exported as.
                self._fixed_matrix[index] = np.asarray(node["matrix"], dtype=float).reshape(4, 4).T
            self.rest_translation[index] = node.get("translation", [0, 0, 0])
            self.rest_rotation[index] = node.get("rotation", [0, 0, 0, 1])
            self.rest_scale[index] = node.get("scale", [1, 1, 1])
        self.order = self._topological_order()
        self.materials = [self._material(m) for m in document.get("materials", [])]
        if not self.materials:
            self.materials = [Material("default", np.array([0.8, 0.8, 0.8, 1.0]))]
        self.skins = [Skin(name=s.get("name", ""), joints=np.asarray(s["joints"], dtype=int),
                           inverse_bind=(self._accessor(s["inverseBindMatrices"])
                                         if "inverseBindMatrices" in s
                                         else np.tile(np.eye(4), (len(s["joints"]), 1, 1))))
                      for s in document.get("skins", [])]
        self.primitives = self._primitives()
        self.clips = {}
        for animation in document.get("animations", []):
            clip = self._clip(animation)
            self.clips[clip.name] = clip

    # -- loading --------------------------------------------------------------

    @classmethod
    def load(cls, source):
        """Read a ``.glb`` from a path or from its bytes."""
        data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
        if len(data) < 20:
            raise GltfError("Too short to be a GLB file.")
        magic, version, length = struct.unpack_from("<III", data, 0)
        if magic != _GLB_MAGIC:
            raise GltfError("Not a binary glTF (.glb) file.")
        if version != 2:
            raise GltfError(f"glTF version {version} is not supported (2 is).")
        offset, document, binary = 12, None, b""
        while offset + 8 <= min(length, len(data)):
            chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8: offset + 8 + chunk_length]
            if chunk_type == _JSON_CHUNK:
                document = json.loads(chunk)
            elif chunk_type == _BIN_CHUNK:
                binary = bytes(chunk)
            offset += 8 + chunk_length
        if document is None:
            raise GltfError("The GLB has no JSON chunk.")
        required = set(document.get("extensionsRequired", []))
        if required:
            raise GltfError(f"The GLB requires extensions this reader lacks: {sorted(required)}.")
        return cls(document, binary)

    def _accessor(self, index):
        accessor = self._doc["accessors"][index]
        dtype = np.dtype(_COMPONENT_TYPES[accessor["componentType"]])
        width = _COMPONENT_COUNTS[accessor["type"]]
        count = accessor["count"]
        if "sparse" in accessor:
            raise GltfError("Sparse accessors are not supported.")
        if "bufferView" not in accessor:
            array = np.zeros((count, width), dtype=dtype)
        else:
            view = self._doc["bufferViews"][accessor["bufferView"]]
            if view.get("buffer", 0) != 0:
                raise GltfError("Only the GLB's own binary buffer is supported.")
            start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            stride = view.get("byteStride") or dtype.itemsize * width
            array = np.ndarray(shape=(count, width), dtype=dtype, buffer=self._bin,
                               offset=start, strides=(stride, dtype.itemsize)).copy()
        if accessor.get("normalized"):
            info = np.iinfo(dtype)
            array = np.maximum(array.astype(np.float64) / info.max, -1.0)
        if accessor["type"] == "MAT4":
            # Stored column-major.
            array = array.reshape(count, 4, 4).transpose(0, 2, 1).astype(np.float64)
        return array

    def _material(self, raw):
        pbr = raw.get("pbrMetallicRoughness", {})
        return Material(name=raw.get("name", ""),
                        base_color=np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), dtype=float),
                        metallic=float(pbr.get("metallicFactor", 1.0)),
                        roughness=float(pbr.get("roughnessFactor", 1.0)),
                        double_sided=bool(raw.get("doubleSided", False)))

    def _primitives(self):
        found = []
        for node_index, node in enumerate(self._doc.get("nodes", [])):
            if "mesh" not in node:
                continue
            mesh = self._doc["meshes"][node["mesh"]]
            for raw in mesh.get("primitives", []):
                if raw.get("mode", _TRIANGLES) != _TRIANGLES:
                    continue
                attributes = raw["attributes"]
                positions = self._accessor(attributes["POSITION"]).astype(np.float32)
                if "indices" in raw:
                    indices = self._accessor(raw["indices"]).reshape(-1, 3).astype(np.int32)
                else:
                    indices = np.arange(len(positions), dtype=np.int32).reshape(-1, 3)
                normals = (self._accessor(attributes["NORMAL"]).astype(np.float32)
                           if "NORMAL" in attributes else _face_normals(positions, indices))
                primitive = Primitive(positions=positions, normals=normals, indices=indices,
                                      material=int(raw.get("material", 0)), node=node_index)
                if "skin" in node and "JOINTS_0" in attributes and "WEIGHTS_0" in attributes:
                    primitive.skin = int(node["skin"])
                    primitive.joints = self._accessor(attributes["JOINTS_0"]).astype(np.int64)
                    weights = self._accessor(attributes["WEIGHTS_0"]).astype(np.float64)
                    total = weights.sum(axis=1, keepdims=True)
                    primitive.weights = (weights / np.where(total == 0, 1, total)).astype(np.float32)
                found.append(primitive)
        return found

    def _clip(self, raw):
        clip = Clip(name=raw.get("name", f"clip{len(self.clips)}"))
        for channel in raw.get("channels", []):
            target = channel["target"]
            if "node" not in target or target["path"] not in ("translation", "rotation", "scale"):
                continue
            sampler = raw["samplers"][channel["sampler"]]
            times = self._accessor(sampler["input"]).reshape(-1).astype(np.float64)
            values = self._accessor(sampler["output"]).astype(np.float64)
            interpolation = sampler.get("interpolation", "LINEAR")
            if interpolation == "CUBICSPLINE":
                values = values.reshape(len(times), 3, -1)
            clip.channels.append(Channel(node=int(target["node"]), path=target["path"],
                                         interpolation=interpolation, times=times, values=values))
        return clip

    def _topological_order(self):
        order, seen = [], set()

        def visit(index):
            if index in seen:
                return
            parent = self.parents[index]
            if parent >= 0:
                visit(parent)
            seen.add(index)
            order.append(index)

        for index in range(len(self.parents)):
            visit(index)
        return order

    # -- evaluating -----------------------------------------------------------

    def node_index(self, name):
        return self.node_names.index(name)

    def local_transforms(self, clip=None, time=0.0):
        """Translation, rotation and scale of every node with ``clip`` applied at ``time``.

        Starts from the exported transforms and replaces only what the clip
        animates; ``clip=None`` gives the exported (bind) transforms.
        """
        translation = self.rest_translation.copy()
        rotation = self.rest_rotation.copy()
        scale = self.rest_scale.copy()
        if clip is not None:
            if isinstance(clip, str):
                if clip not in self.clips:
                    raise KeyError(f"The model has no clip named {clip!r}.")
                clip = self.clips[clip]
            target = {"translation": translation, "rotation": rotation, "scale": scale}
            for channel in clip.channels:
                target[channel.path][channel.node] = sample_channel(channel, time)
        return translation, rotation, scale

    def world_matrices(self, translation, rotation, scale):
        local = trs_matrices(translation, rotation, scale)
        for index, matrix in self._fixed_matrix.items():
            local[index] = matrix
        world = np.empty_like(local)
        for index in self.order:
            parent = self.parents[index]
            world[index] = local[index] if parent < 0 else world[parent] @ local[index]
        return world

    def pose(self, clip=None, time=0.0):
        """World matrices of every node and the skinning matrices of every skin.

        Returns
        ----------
        (world, joint_matrices): ((n, 4, 4) array, list of (J, 4, 4) arrays)
        """
        world = self.world_matrices(*self.local_transforms(clip, time))
        joints = [world[skin.joints] @ skin.inverse_bind for skin in self.skins]
        return world, joints

    def skin_vertices(self, primitive, world, joint_matrices):
        """World-space positions and normals of one primitive in this pose."""
        if primitive.skin is None:
            matrix = world[primitive.node]
            positions = primitive.positions @ matrix[:3, :3].T + matrix[:3, 3]
            normals = primitive.normals @ np.linalg.inv(matrix[:3, :3])
        else:
            matrices = joint_matrices[primitive.skin]
            # One blended matrix per vertex: sum_k w_k * M[j_k]. The upper 3x4
            # is all that is needed.
            blended = np.einsum("nk,nkij->nij", primitive.weights,
                                matrices[primitive.joints][:, :, :3, :])
            positions = (np.einsum("nij,nj->ni", blended[:, :, :3], primitive.positions)
                         + blended[:, :, 3])
            # The rig only rotates and (near-)uniformly scales, so the blended
            # linear part transforms normals as well as positions once they are
            # renormalised.
            normals = np.einsum("nij,nj->ni", blended[:, :, :3], primitive.normals)
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(length == 0, 1, length)
        return positions.astype(np.float32), normals.astype(np.float32)


def sample_channel(channel, time):
    """The value of one animation channel at ``time``, clamped to its keyframes."""
    times = channel.times
    values = channel.values
    cubic = channel.interpolation == "CUBICSPLINE"
    if len(times) == 1 or time <= times[0]:
        return values[0, 1] if cubic else values[0]
    if time >= times[-1]:
        return values[-1, 1] if cubic else values[-1]
    upper = int(np.searchsorted(times, time, side="right"))
    lower = upper - 1
    span = times[upper] - times[lower]
    alpha = 0.0 if span <= 0 else (time - times[lower]) / span
    if channel.interpolation == "STEP":
        return values[lower]
    if cubic:
        t2, t3 = alpha * alpha, alpha * alpha * alpha
        p0, m0 = values[lower, 1], values[lower, 2] * span
        p1, m1 = values[upper, 1], values[upper, 0] * span
        value = ((2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + alpha) * m0
                 + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * m1)
        return quat_normalize(value) if channel.path == "rotation" else value
    if channel.path == "rotation":
        return quat_slerp(values[lower], values[upper], alpha)
    return values[lower] + alpha * (values[upper] - values[lower])


def _face_normals(positions, indices):
    normals = np.zeros_like(positions, dtype=np.float64)
    a, b, c = (positions[indices[:, k]] for k in range(3))
    face = np.cross(b - a, c - a)
    for k in range(3):
        np.add.at(normals, indices[:, k], face)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.where(length == 0, 1, length)).astype(np.float32)


__all__ = ["GltfModel", "GltfError", "Material", "Primitive", "Skin", "Channel", "Clip",
           "sample_channel", "quat_slerp", "quat_to_matrix", "trs_matrices"]
