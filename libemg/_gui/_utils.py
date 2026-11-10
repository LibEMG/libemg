from PIL import Image
import numpy as np
import dearpygui.dearpygui as dpg
import cv2

class Media:
    def __init__(self):
        pass

    def from_file(self, location, fps=24):
        self.type = location.split(".")[-1]
        self.fps = fps
        if self.type == "mp4":
            self.import_video(location)
        elif self.type in ["png", "jpg", "bmp"]:
            self.import_picture(location)
        elif self.type == "gif":
            self.import_gif(location)    

    def import_picture(self, location):
        self.file_content = Image.open(location)
        self._invalidate_texture_cache()

    def import_gif(self, location):
        self.file_content = Image.open(location)
        self.frame = 0
        self.file_content.seek(self.frame)
        self.n_frames = self.file_content.n_frames
        self.frame_times = np.linspace(0, self.n_frames/self.fps, int(self.n_frames))
        self._invalidate_texture_cache()

    def import_video(self, location):
        # get video capture ready
        self.video_capture = cv2.VideoCapture(location)
        self.n_frames = self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frame_times = np.linspace(0, self.n_frames/self.fps, int(self.n_frames))
        # get first frame
        self.frame = 0
        _, cv2_image  = self.video_capture.read()
        cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGBA)
        self.file_content =  Image.fromarray(cv2_image)
        self._invalidate_texture_cache()

    def from_numpy(self, numpy_array):
        self.file_content = Image.fromarray(numpy_array)
        self.type = "png"
        self._invalidate_texture_cache()
    
    def reset(self):
        if self.type == "gif":
            self.frame = 0
            self.file_content.seek(self.frame)
            self._invalidate_texture_cache()
        if self.type == "mp4":
            # CAP_PROP_POS_FRAMES is the decoder's read position;
            # CAP_PROP_FRAME_COUNT (which used to be set here) is the read-only
            # length of the clip, so the rewind never actually happened. The
            # sequential fast path in advance_to tracks the decoder with
            # self.frame, so this has to be a real seek.
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.frame = 0
            # Pull frame 0 back out so the object is left in exactly the state
            # import_video leaves it in: file_content holds frame 0 and the
            # decoder is parked on frame 1. Without this read the caller would
            # keep showing the last frame of the previous playthrough and every
            # frame after it would be off by one.
            ret, cv2_image = self.video_capture.read()
            if ret:
                cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGBA)
                self.file_content = Image.fromarray(cv2_image)
            self._invalidate_texture_cache()

    def advance(self):
        assert hasattr(self, "file_content")
        if self.type == "gif":
            if self.frame + 1 < self.file_content.n_frames:
                self.frame += 1
                self.file_content.seek(self.frame)
            else:
                print("End of gif reached.")
        if self.type == "mp4":
            # print(self.frame)
            self.frame += 1
            ret, cv2_image  = self.video_capture.read()
            if not ret:
                print("End of video reached")
            else:
                cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGBA)
                self.file_content =  Image.fromarray(cv2_image)
    
    def advance_to(self, play_time):
        if not hasattr(self, "frame_times"):
            return
        # find the closest time. Frames are evenly spaced at 1/fps, so the index
        # is arithmetic - the old np.abs(frame_times - play_time).argmin()
        # scanned every frame time of the clip on every rendered frame.
        last_frame = max(0, int(self.n_frames) - 1)
        closest_frame = int(round(play_time * self.fps))
        closest_frame = min(max(closest_frame, 0), last_frame)
        if self.type == "gif":
            if closest_frame < self.file_content.n_frames:
                self.frame = closest_frame
                self.file_content.seek(self.frame)
        if self.type == "mp4":
            if closest_frame == self.frame:
                # Already showing this frame, so there is nothing to decode.
                return
            if closest_frame != self.frame + 1:
                # Only seek when the frame cannot be reached by reading forward
                # once: a backward jump, or a skip of more than one frame. A
                # seek sends the decoder back to a keyframe and re-decodes
                # forward from there, which is what made playback pay for a
                # keyframe seek on every single frame.
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, closest_frame)
            ret, cv2_image  = self.video_capture.read()
            if not ret:
                print("End of video reached")
                # Nothing was decoded, so re-derive self.frame from where the
                # decoder actually stopped (the frame still being held is the
                # one before its next read position). Leaving self.frame stale
                # would let the sequential fast path above hand out a frame that
                # was never decoded.
                position = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                self.frame = max(0, position - 1)
            else:
                self.frame = closest_frame
                cv2_image = cv2.cvtColor(cv2_image,cv2.COLOR_BGR2RGBA)
                self.file_content =  Image.fromarray(cv2_image)

    def _invalidate_texture_cache(self):
        """Forget the cached texture, for when file_content has been replaced."""
        self._tex_cache_key = None
        self._tex_cache_value = None

    def get_dpg_formatted_texture(self, width, height, grayscale=False):
        # Rasterising is resize -> convert -> float32 -> divide, ~10.8 ms and
        # ~11 MB at 720x480, and screen guided training asks for a texture once
        # per rendered frame. The cache key carries self.frame, so a still image
        # (which has no frame) produces a constant key and only rasterises once,
        # while a gif/mp4 gets a new key per frame and correctly re-rasterises -
        # for those every frame really is a new image.
        # Only the single most recent entry is kept. That is enough to remove
        # the repeated-call cost for stills and for a frame that gets requested
        # twice, and unlike a multi-entry cache it cannot grow frame by frame
        # while a long video plays.
        cache_key = (self.type, getattr(self, "frame", None), width, height, grayscale)
        if getattr(self, "_tex_cache_key", None) == cache_key:
            # Handed out by reference on purpose: dpg's raw texture keeps
            # reading the very buffer it was given, so returning the same array
            # object is what lets it keep working. Callers must not mutate it.
            return self._tex_cache_value
        dpg_img = self.file_content.resize((width, height))
        if grayscale:
            dpg_img = dpg_img.convert("L")
        dpg_img = dpg_img.convert("RGBA")
        dpg_img = np.asarray(dpg_img, dtype=np.float32).ravel()
        dpg_img = np.true_divide(dpg_img, 255.0)
        self._tex_cache_key = cache_key
        self._tex_cache_value = dpg_img
        return dpg_img

# One texture registry shared by the whole process. dpg.texture_registry() is
# add_texture_registry + push_container_stack, so using it as a context manager
# mints a brand new registry item with a fresh uuid on every call - set_texture
# runs once per rendered frame, so that leaked thousands of orphan registry
# items per session.
TEXTURE_REGISTRY_TAG = "__libemg_texture_registry"

def get_texture_registry():
    """Return the tag of the shared texture registry, creating it on first use."""
    if not dpg.does_item_exist(TEXTURE_REGISTRY_TAG):
        if dpg.does_alias_exist(TEXTURE_REGISTRY_TAG):
            # An alias left behind by a deleted registry would make the add
            # below raise, so clear it first.
            dpg.remove_alias(TEXTURE_REGISTRY_TAG)
        dpg.add_texture_registry(show=False, tag=TEXTURE_REGISTRY_TAG)
    return TEXTURE_REGISTRY_TAG

def set_texture(tag, texture, width, height, format=dpg.mvFormat_Float_rgba):
    # Updating an existing texture needs no container at all, and this is the
    # path taken on every frame after the first, so take it first and entirely
    # outside any registry context.
    if dpg.does_item_exist(tag):
        dpg.set_value(tag, value=texture)
        return
    # Only a newly created texture needs a registry to live in; parent it to the
    # shared one instead of pushing a new container.
    dpg.add_raw_texture(width=width,
                        height=height,
                        default_value=texture,
                        tag=tag,
                        format=format,
                        parent=get_texture_registry())
