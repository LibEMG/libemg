import numpy as np

from libemg._gui._utils import Media


def test_get_dpg_formatted_texture_returns_normalized_rgba_values():
    image = np.array(
        [
            [[0, 64, 128], [255, 192, 128]],
            [[255, 255, 255], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    media = Media()
    media.from_numpy(image)

    texture = media.get_dpg_formatted_texture(width=2, height=2)

    expected = np.array(
        [
            0, 64, 128, 255,
            255, 192, 128, 255,
            255, 255, 255, 255,
            0, 0, 0, 255,
        ],
        dtype=np.float32,
    ) / 255
    np.testing.assert_allclose(texture, expected)
