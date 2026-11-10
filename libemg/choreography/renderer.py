"""Offscreen rendering of the skinned hand into an RGBA buffer for Dear PyGui.

Dear PyGui draws textures; it has no 3D scene, no skinning and no way to share
its own GPU device. So the hand is drawn here, into a plain float32 RGBA array,
and handed to a DPG raw texture. The rasteriser runs on the CPU -- skinning in
numpy, the triangle loop compiled by numba with the GIL released -- because the
project ships no OpenGL/Vulkan binding and a CPU path has no context to tie to a
thread, no driver to fail on a lab machine, and nothing to leak at shutdown.

At the default 480x480 with 2x supersampling a frame takes on the order of
10-20 ms, rendered on :class:`RenderWorker`'s thread, never on the GUI thread or
the collection logger's.

Lighting is a hemisphere ambient plus a key, fill and rim light with
Blinn-Phong highlights, applied per vertex (the hand is dense enough that this
is indistinguishable from per-pixel shading at this size). Material colours are
the GLB's linear ``baseColorFactor`` values, shaded in linear space and encoded
to sRGB at the end.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from .gltf import GltfModel

try:  # numba ships with librosa, a hard dependency; the fallback is for safety only.
    from numba import njit as _njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

    def _njit(*_args, **_kwargs):
        def wrap(function):
            return function
        return wrap


@_njit(nogil=True, cache=True)
def _rasterize(sx, sy, sz, colors, triangles, color_buffer, depth_buffer):
    """Z-buffered triangle fill with colours interpolated across each triangle.

    ``sx``/``sy`` are pixel coordinates (y down), ``sz`` a depth where smaller is
    nearer. Pixel centres are sampled at +0.5. Both windings are drawn: every
    material on this asset is double-sided, and the depth test hides the back.
    """
    height, width = depth_buffer.shape
    for t in range(triangles.shape[0]):
        i0 = triangles[t, 0]
        i1 = triangles[t, 1]
        i2 = triangles[t, 2]
        x0 = sx[i0]
        y0 = sy[i0]
        x1 = sx[i1]
        y1 = sy[i1]
        x2 = sx[i2]
        y2 = sy[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area > -1e-12 and area < 1e-12:
            continue
        min_x = max(int(math.floor(min(x0, min(x1, x2)))), 0)
        max_x = min(int(math.ceil(max(x0, max(x1, x2)))), width - 1)
        min_y = max(int(math.floor(min(y0, min(y1, y2)))), 0)
        max_y = min(int(math.ceil(max(y0, max(y1, y2)))), height - 1)
        if min_x > max_x or min_y > max_y:
            continue
        inverse = 1.0 / area
        z0 = sz[i0]
        z1 = sz[i1]
        z2 = sz[i2]
        for py in range(min_y, max_y + 1):
            fy = py + 0.5
            for px in range(min_x, max_x + 1):
                fx = px + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) * inverse
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) * inverse
                w2 = 1.0 - w0 - w1
                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z < depth_buffer[py, px]:
                    depth_buffer[py, px] = z
                    for c in range(3):
                        color_buffer[py, px, c] = (w0 * colors[i0, c] + w1 * colors[i1, c]
                                                   + w2 * colors[i2, c])


@_njit(nogil=True, cache=True)
def _clear(background_rows, color_buffer, depth_buffer):
    """Fill the colour buffer with the backdrop (one colour per row) and reset depth."""
    height, width = depth_buffer.shape
    for y in range(height):
        for x in range(width):
            depth_buffer[y, x] = np.inf
            for c in range(3):
                color_buffer[y, x, c] = background_rows[y, c]


@_njit(nogil=True, cache=True)
def _resolve(color_buffer, ss, out):
    """Average each ``ss`` x ``ss`` block into one RGBA pixel of ``out`` (flat)."""
    height = color_buffer.shape[0] // ss
    width = color_buffer.shape[1] // ss
    scale = 1.0 / (ss * ss)
    for y in range(height):
        for x in range(width):
            base = (y * width + x) * 4
            for c in range(3):
                total = 0.0
                for dy in range(ss):
                    for dx in range(ss):
                        total += color_buffer[y * ss + dy, x * ss + dx, c]
                out[base + c] = total * scale
            out[base + 3] = 1.0


@dataclass
class ViewSettings:
    """Where the camera sits relative to the hand.

    The defaults show a right hand from behind and to the dorsal side, slightly
    above: thumb up, palm facing screen-left, fingers reaching right and into
    the screen -- the participant's own view of their hand, turned enough that
    finger and wrist motion read in profile.
    """

    yaw_degrees: float = -32.0      # about world +Y (thumb axis); 0 is behind the forearm, negative toward the palm
    pitch_degrees: float = 16.0     # above the hand
    fov_degrees: float = 30.0
    margin: float = 1.06            # framing slack around the motion envelope
    background_top: tuple = (0.30, 0.31, 0.33)       # sRGB, neutral grey
    background_bottom: tuple = (0.17, 0.175, 0.185)


def _look_at(eye, target, up):
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    view = np.eye(4)
    view[0, :3], view[1, :3], view[2, :3] = right, true_up, -forward
    view[:3, 3] = -view[:3, :3] @ eye
    return view, right, true_up, forward


def _perspective(fov_degrees, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_degrees) / 2)
    projection = np.zeros((4, 4))
    projection[0, 0] = f / aspect
    projection[1, 1] = f
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2 * far * near / (near - far)
    projection[3, 2] = -1.0
    return projection


def _linear_to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear,
                    1.055 * np.power(linear, 1 / 2.4) - 0.055)


class HandRenderer:
    """Draws a :class:`GltfModel` in a given pose into an RGBA float32 image.

    Parameters
    ----------
    model: GltfModel
    width, height: int
        Output size in pixels.
    supersample: int
        Render at this multiple of the output size and average down, for
        anti-aliased edges.
    view: ViewSettings, optional
    framing_poses: list of (clip, time), optional
        Poses whose combined extent the camera is fitted to, so no motion leaves
        the frame. Defaults to every clip at its target time.
    linear_motion: bool
        Re-time each clip's ramps so the hand moves linearly in time, matching
        the ``position`` target (see :class:`~libemg.choreography.animation.MotionProfile`).
        :meth:`render` then takes *nominal* clip times.
    """

    def __init__(self, model, width=480, height=480, supersample=2, view=None,
                 framing_poses=None, target_time=1.0, linear_motion=True):
        self.model = model
        self.profile = None
        if linear_motion:
            from .animation import MotionProfile
            self.profile = MotionProfile.measure(model)
        self.view = view or ViewSettings()
        self.supersample = max(1, int(supersample))
        self.width = self.height = 0
        self._background = None
        self._prepare_geometry()
        poses = framing_poses or [(None, 0.0)] + [(name, target_time) for name in model.clips]
        self._fit_camera(poses)
        self.resize(width, height)
        self.closed = False

    # -- set-up ---------------------------------------------------------------

    def _prepare_geometry(self):
        offsets, triangles, colors, gloss = [], [], [], []
        count = 0
        for primitive in self.model.primitives:
            offsets.append(count)
            triangles.append(primitive.indices + count)
            material = self.model.materials[min(primitive.material, len(self.model.materials) - 1)]
            n = len(primitive.positions)
            colors.append(np.tile(material.base_color[:3], (n, 1)))
            gloss.append(np.full(n, material.roughness))
            count += n
        self._triangles = np.ascontiguousarray(np.vstack(triangles), dtype=np.int32)
        self._base_color = np.vstack(colors)
        self._roughness = np.concatenate(gloss)
        self.vertex_count = count
        self.triangle_count = len(self._triangles)

    def motion_record(self):
        """How poses are timed, for the run record."""
        if self.profile is None:
            return {"motion_profile": "authored", "note": "clip easing as exported from Blender"}
        return self.profile.record()

    def skinned(self, clip=None, time=0.0):
        """Every vertex of the model in this pose, world space: (positions, normals)."""
        world, joints = self.model.pose(clip, time)
        positions, normals = [], []
        for primitive in self.model.primitives:
            p, n = self.model.skin_vertices(primitive, world, joints)
            positions.append(p)
            normals.append(n)
        return np.vstack(positions).astype(np.float64), np.vstack(normals).astype(np.float64)

    def _fit_camera(self, poses):
        points = np.vstack([self.skinned(clip, t)[0] for clip, t in poses])
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        radius = float(np.linalg.norm(points - center, axis=1).max())
        yaw, pitch = math.radians(self.view.yaw_degrees), math.radians(self.view.pitch_degrees)
        # In the exported (glTF, Y-up) frame the fingers point down -Z, the
        # thumb up +Y and the palm toward -X; +Z is behind the forearm. A
        # negative yaw swings the camera round toward the palm side.
        direction = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch),
                              math.cos(yaw) * math.cos(pitch)])
        up = np.array([0.0, 1.0, 0.0])
        _, right, true_up, forward = _look_at(center + direction, center, up)
        # Frame what the motions actually sweep across the screen, rather than
        # a sphere round them: the envelope is long and thin, and a sphere
        # leaves the hand small in the middle of a lot of backdrop.
        across, along, depth = points @ right, points @ true_up, points @ forward
        middle = (right * (across.min() + across.max()) / 2
                  + true_up * (along.min() + along.max()) / 2
                  + forward * (depth.min() + depth.max()) / 2)
        half_extent = max(across.max() - across.min(), along.max() - along.min()) / 2
        half_depth = (depth.max() - depth.min()) / 2
        tangent = math.tan(math.radians(self.view.fov_degrees) / 2)
        distance = half_extent * self.view.margin / tangent + half_depth
        self.eye = middle + direction * distance
        self.target = middle
        self.near = max(1e-3, distance - radius * 1.5)
        self.far = distance + radius * 1.5
        self._view_matrix, self._right, self._up, self._forward = _look_at(
            self.eye, self.target, up)
        # Lights in camera terms so the framing and the lighting move together.
        right, up, toward_viewer = self._right, self._up, -self._forward
        self._lights = [
            # (direction toward the light, linear colour)
            (_unit(0.55 * right + 0.75 * up + 0.65 * toward_viewer), np.array([1.00, 0.97, 0.92]) * 1.05),
            (_unit(-0.75 * right + 0.15 * up + 0.55 * toward_viewer), np.array([0.62, 0.68, 0.80]) * 0.35),
            (_unit(-0.20 * right + 0.45 * up - 0.95 * toward_viewer), np.array([0.90, 0.92, 1.00]) * 0.55),
        ]

    def resize(self, width, height):
        """Change the output size. Takes effect on the next :meth:`render`."""
        width, height = int(width), int(height)
        if width < 16 or height < 16:
            raise ValueError("The hand viewport must be at least 16x16 pixels.")
        if (width, height) == (self.width, self.height):
            return
        self.width, self.height = width, height
        ss = self.supersample
        self._projection = _perspective(self.view.fov_degrees, width / height, self.near, self.far)
        rows = np.linspace(0.0, 1.0, height * ss)[:, None]
        top, bottom = np.array(self.view.background_top), np.array(self.view.background_bottom)
        self._background = np.ascontiguousarray(top[None, :] * (1 - rows) + bottom[None, :] * rows,
                                                dtype=np.float32)
        # Work buffers reused frame to frame; only the output is handed out.
        self._color = np.empty((height * ss, width * ss, 3), dtype=np.float32)
        self._depth = np.empty((height * ss, width * ss), dtype=np.float64)

    # -- drawing --------------------------------------------------------------

    def _shade(self, positions, normals):
        to_eye = self.eye[None, :] - positions
        to_eye /= np.linalg.norm(to_eye, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", normals, to_eye)
        normals = np.where(facing[:, None] < 0, -normals, normals)    # double-sided
        up = normals[:, 1]
        sky, ground = np.array([0.42, 0.44, 0.48]), np.array([0.16, 0.15, 0.14])
        light = (sky[None, :] * (0.5 + 0.5 * up[:, None])
                 + ground[None, :] * (0.5 - 0.5 * up[:, None])) * 0.55
        specular = np.zeros_like(positions)
        roughness = self._roughness
        shininess = np.clip(2.0 / np.maximum(roughness, 0.05) ** 4 - 2.0, 4.0, 512.0)
        strength = 0.35 * (1.0 - roughness) + 0.04
        for direction, color in self._lights:
            lambert = np.clip(normals @ direction, 0.0, None)
            light += lambert[:, None] * color[None, :]
            half = _unit_rows(direction[None, :] + to_eye)
            highlight = np.clip(np.einsum("ij,ij->i", normals, half), 0.0, None) ** shininess
            specular += (highlight * strength * (lambert > 0))[:, None] * color[None, :]
        # A soft rim so the silhouette separates from the backdrop.
        rim = (1.0 - np.clip(np.einsum("ij,ij->i", normals, to_eye), 0.0, 1.0)) ** 3
        linear = self._base_color * light + specular + 0.10 * rim[:, None]
        return _linear_to_srgb(linear).astype(np.float32)

    def render(self, clip=None, time=0.0, out=None):
        """Draw the hand posed by ``clip`` at ``time``.

        Parameters
        ----------
        out: np.ndarray, optional
            A float32 array of ``height * width * 4`` to write into, so a caller
            can recycle buffers. Allocated when omitted.

        Returns
        ----------
        np.ndarray
            ``(height * width * 4,)`` float32 RGBA in [0, 1], rows top to
            bottom -- the layout ``dpg.add_raw_texture`` with
            ``mvFormat_Float_rgba`` expects.
        """
        if self.closed:
            raise RuntimeError("This renderer has been closed.")
        ss = self.supersample
        full_w, full_h = self.width * ss, self.height * ss
        if self.profile is not None:
            time = self.profile.clip_time(clip, time)
        positions, normals = self.skinned(clip, time)
        colors = self._shade(positions, normals)
        homogeneous = np.hstack([positions, np.ones((len(positions), 1))])
        clip_space = homogeneous @ (self._projection @ self._view_matrix).T
        w = clip_space[:, 3]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        ndc = clip_space[:, :3] / w[:, None]
        sx = ((ndc[:, 0] + 1.0) * 0.5 * full_w).astype(np.float64)
        sy = ((1.0 - ndc[:, 1]) * 0.5 * full_h).astype(np.float64)
        sz = ndc[:, 2].astype(np.float64)
        _clear(self._background, self._color, self._depth)
        _rasterize(sx, sy, sz, colors, self._triangles, self._color, self._depth)
        if out is None or out.size != self.width * self.height * 4 or out.dtype != np.float32:
            out = np.empty(self.width * self.height * 4, dtype=np.float32)
        _resolve(self._color, ss, out)
        return out

    def project(self, points):
        """Pixel coordinates (x right, y down) of world points, at the output size."""
        points = np.atleast_2d(points)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        clip_space = homogeneous @ (self._projection @ self._view_matrix).T
        ndc = clip_space[:, :3] / clip_space[:, 3:4]
        return np.column_stack([(ndc[:, 0] + 1) * 0.5 * self.width,
                                (1 - ndc[:, 1]) * 0.5 * self.height])

    def close(self):
        """Drop the buffers. There is no GPU state to release on this path."""
        self.closed = True
        self._background = self._color = self._depth = None


def _unit(vector):
    vector = np.asarray(vector, dtype=float)
    return vector / np.linalg.norm(vector)


def _unit_rows(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


# -- the render thread ----------------------------------------------------------

@dataclass
class Frame:
    """One finished render, and the scheduler moment it shows."""

    frame_id: int
    buffer: np.ndarray
    width: int
    height: int
    clip: str | None
    clip_time: float
    request_monotonic: float
    render_seconds: float
    slot: int


class RenderWorker:
    """Renders on its own thread, always the newest pose asked for.

    The GUI thread calls :meth:`request` once a frame with the pose the
    scheduler wants shown, and :meth:`take` to collect whatever has finished.
    Requests are latest-wins, so a slow frame never builds a backlog: the
    worker skips straight to the newest pose.

    Buffers are recycled through a ring of three. DearPyGui's raw texture keeps
    reading the array it was given, so the worker never writes into the buffer
    that is on screen or the one waiting to be shown.

    Parameters
    ----------
    factory: callable
        Builds the :class:`HandRenderer` -- called on the worker thread, so
        loading the model and compiling the rasteriser never block the GUI.
    """

    RING = 3

    def __init__(self, factory, width=480, height=480):
        self._factory = factory
        self._condition = threading.Condition()
        self._request = None
        self._pending = None
        self._size = (int(width), int(height))
        self._rings = {}
        self._displayed = None          # (size, slot) currently handed to DPG
        self._next_id = 0
        self._closing = False
        self.error = None
        self.ready = threading.Event()
        self.renderer = None
        self._thread = threading.Thread(target=self._run, name="choreography-render", daemon=True)
        self._thread.start()

    def request(self, clip, clip_time, request_monotonic):
        with self._condition:
            self._request = (clip, float(clip_time), float(request_monotonic))
            self._condition.notify()

    def resize(self, width, height):
        with self._condition:
            self._size = (int(width), int(height))
            self._condition.notify()

    @property
    def size(self):
        return self._size

    def take(self):
        """The newest finished frame not yet taken, or None. Marks it as displayed."""
        with self._condition:
            frame, self._pending = self._pending, None
            if frame is not None:
                self._displayed = ((frame.width, frame.height), frame.slot)
            return frame

    def close(self, timeout=2.0):
        with self._condition:
            self._closing = True
            self._condition.notify()
        self._thread.join(timeout)
        if self.renderer is not None:
            self.renderer.close()
        self._rings.clear()

    @property
    def alive(self):
        return self._thread.is_alive()

    def _slot_for(self, size):
        ring = self._rings.setdefault(size, [np.zeros(size[0] * size[1] * 4, dtype=np.float32)
                                             for _ in range(self.RING)])
        busy = set()
        if self._displayed and self._displayed[0] == size:
            busy.add(self._displayed[1])
        if self._pending is not None and (self._pending.width, self._pending.height) == size:
            busy.add(self._pending.slot)
        slot = next(i for i in range(self.RING) if i not in busy)
        return slot, ring[slot]

    def _run(self):
        try:
            self.renderer = self._factory()
            # The first call compiles the rasteriser; do it before anyone waits on a frame.
            self.renderer.resize(*self._size)
            self.renderer.render(None, 0.0)
        except Exception as error:      # noqa: BLE001 - surfaced through .error
            self.error = error
            self.ready.set()
            return
        self.ready.set()
        while True:
            with self._condition:
                while self._request is None and not self._closing:
                    self._condition.wait(0.25)
                if self._closing:
                    return
                request, self._request = self._request, None
                size = self._size
                slot, buffer = self._slot_for(size)
            clip, clip_time, request_monotonic = request
            started = time.perf_counter()
            try:
                self.renderer.resize(*size)
                self.renderer.render(clip, clip_time, out=buffer)
            except Exception as error:  # noqa: BLE001 - surfaced through .error
                self.error = error
                return
            frame = Frame(frame_id=self._next_id, buffer=buffer, width=size[0], height=size[1],
                          clip=clip, clip_time=clip_time, request_monotonic=request_monotonic,
                          render_seconds=time.perf_counter() - started, slot=slot)
            self._next_id += 1
            with self._condition:
                self._pending = frame


__all__ = ["HandRenderer", "RenderWorker", "Frame", "ViewSettings", "HAVE_NUMBA"]
