import numpy as np
from rotorpy.guidance.pixels import target_centroid, target_pixel_count

def test_empty_image_has_no_target():
    image = np.zeros((4, 4, 3))
    assert target_pixel_count(image) == 0
    assert target_centroid(image) is None

def test_red_threshold_and_centroid():
    image = np.zeros((4, 5, 3))
    image[1, 2] = [0.51, 0.5, 0.5]
    image[1, 3] = [0.5, 0.0, 0.0]       # red == 0.5 does not count
    image[2, 2] = [0.9, 0.9, 0.0]       # red is not strictly greater than green
    assert target_pixel_count(image) == 1
    assert np.allclose(target_centroid(image), [2.0, 1.0])

def test_centroid_is_mean_of_every_target_pixel():
    image = np.zeros((4, 5, 3))
    image[0, 0] = [0.8, 0.0, 0.0]
    image[2, 4] = [0.9, 0.1, 0.1]
    assert target_pixel_count(image) == 2
    assert np.allclose(target_centroid(image), [2.0, 1.0])

def test_red_equal_blue_is_not_a_target():
    image = np.zeros((3, 3, 3))
    image[1, 1] = [0.8, 0.1, 0.8]  # red == blue, red > 0.5, red > green
    assert target_pixel_count(image) == 0
    assert target_centroid(image) is None
