# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------

import math
import os

import numpy as np
import skimage as sk
from tdb.tdbopenslide import SlideInfo

from . import slide, util
from .util import Time


def mask_percent(np_img):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

    Args:
      np_img: Image as a NumPy array.

    Returns:
      The percentage of the NumPy array that is masked.
    """
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    return mask_percentage


def tissue_percent(np_img):
    """
    Determine the percentage of a NumPy array that is tissue (not masked).

    Args:
      np_img: Image as a NumPy array.

    Returns:
      The percentage of the NumPy array that is tissue.
    """
    return 100 - mask_percent(np_img)


def filter_remove_small_objects(
    np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"
):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.

    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Minimum size of small object to remove.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array (bool, float, or uint8).
    """
    t = Time()

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk.morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (
        (mask_percentage >= overmask_thresh)
        and (min_size >= 1)
        and (avoid_overmask is True)
    ):
        new_min_size = min_size / 2
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d"
            % (mask_percentage, overmask_thresh, min_size, new_min_size)
        )
        rem_sm = filter_remove_small_objects(
            np_img, new_min_size, avoid_overmask, overmask_thresh, output_type
        )
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    util.np_info(np_img, "Remove Small Objs", t.elapsed())
    return np_img


def filter_rgb_to_hsv(np_img, display_np_info=True):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).

    Args:
      np_img: RGB image as a NumPy array.
      display_np_info: If True, display NumPy array info and filter time.

    Returns:
      Image as NumPy array in HSV representation.
    """

    if display_np_info:
        t = Time()
    hsv = sk.color.rgb2hsv(np_img)
    if display_np_info:
        util.np_info(hsv, "RGB to HSV", t.elapsed())
    return hsv


def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):
    """
    Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
    values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
    https://en.wikipedia.org/wiki/HSL_and_HSV

    Args:
      hsv: HSV image as a NumPy array.
      output_type: Type of array to return (float or int).
      display_np_info: If True, display NumPy array info and filter time.

    Returns:
      Hue values (float or int) as a 1-dimensional NumPy array.
    """
    if display_np_info:
        t = Time()
    h = hsv[:, :, 0]
    h = h.flatten()
    if output_type == "int":
        h *= 360
        h = h.astype("int")
    if display_np_info:
        util.np_info(hsv, "HSV to H", t.elapsed())
    return h


def filter_hsv_to_s(hsv):
    """
    Experimental HSV to S (saturation).

    Args:
      hsv:  HSV image as a NumPy array.

    Returns:
      Saturation values as a 1-dimensional NumPy array.
    """
    s = hsv[:, :, 1]
    s = s.flatten()
    return s


def filter_hsv_to_v(hsv):
    """
    Experimental HSV to V (value).

    Args:
      hsv:  HSV image as a NumPy array.

    Returns:
      Value values as a 1-dimensional NumPy array.
    """
    v = hsv[:, :, 2]
    v = v.flatten()
    return v


def filter_green_channel(
    np_img,
    green_thresh=200,
    avoid_overmask=True,
    overmask_thresh=90,
    output_type="bool",
):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin
    and eosin are purplish and pinkish, which do not have much green to them.

    Args:
      np_img: RGB image as a NumPy array.
      green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """
    t = Time()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (
        (mask_percentage >= overmask_thresh)
        and (green_thresh < 255)
        and (avoid_overmask is True)
    ):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d"
            % (mask_percentage, overmask_thresh, green_thresh, new_green_thresh)
        )
        gr_ch_mask = filter_green_channel(
            np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type
        )
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    util.np_info(np_img, "Filter Green Channel", t.elapsed())
    return np_img


def filter_red(
    rgb,
    red_lower_thresh,
    green_upper_thresh,
    blue_upper_thresh,
    output_type="bool",
    display_np_info=False,
):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.

    Args:
      rgb: RGB image as a NumPy array.
      red_lower_thresh: Red channel lower threshold value.
      green_upper_thresh: Green channel upper threshold value.
      blue_upper_thresh: Blue channel upper threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      display_np_info: If True, display NumPy array info and filter time.

    Returns:
      NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        util.np_info(result, "Filter Red", t.elapsed())
    return result


def filter_red_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out red pen marks from a slide.

    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array representing the mask.
    """
    t = Time()
    result = (
        filter_red(
            rgb, red_lower_thresh=150, green_upper_thresh=80, blue_upper_thresh=90
        )
        & filter_red(
            rgb, red_lower_thresh=110, green_upper_thresh=20, blue_upper_thresh=30
        )
        & filter_red(
            rgb, red_lower_thresh=185, green_upper_thresh=65, blue_upper_thresh=105
        )
        & filter_red(
            rgb, red_lower_thresh=195, green_upper_thresh=85, blue_upper_thresh=125
        )
        & filter_red(
            rgb, red_lower_thresh=220, green_upper_thresh=115, blue_upper_thresh=145
        )
        & filter_red(
            rgb, red_lower_thresh=125, green_upper_thresh=40, blue_upper_thresh=70
        )
        & filter_red(
            rgb, red_lower_thresh=200, green_upper_thresh=120, blue_upper_thresh=150
        )
        & filter_red(
            rgb, red_lower_thresh=100, green_upper_thresh=50, blue_upper_thresh=65
        )
        & filter_red(
            rgb, red_lower_thresh=85, green_upper_thresh=25, blue_upper_thresh=45
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    util.np_info(result, "Filter Red Pen", t.elapsed())
    return result


def filter_green(
    rgb,
    red_upper_thresh,
    green_lower_thresh,
    blue_lower_thresh,
    output_type="bool",
    display_np_info=False,
):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.

    Args:
      rgb: RGB image as a NumPy array.
      red_upper_thresh: Red channel upper threshold value.
      green_lower_thresh: Green channel lower threshold value.
      blue_lower_thresh: Blue channel lower threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      display_np_info: If True, display NumPy array info and filter time.

    Returns:
      NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        util.np_info(result, "Filter Green", t.elapsed())
    return result


def filter_green_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out green pen marks from a slide.

    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array representing the mask.
    """
    t = Time()
    result = (
        filter_green(
            rgb, red_upper_thresh=150, green_lower_thresh=160, blue_lower_thresh=140
        )
        & filter_green(
            rgb, red_upper_thresh=70, green_lower_thresh=110, blue_lower_thresh=110
        )
        & filter_green(
            rgb, red_upper_thresh=45, green_lower_thresh=115, blue_lower_thresh=100
        )
        & filter_green(
            rgb, red_upper_thresh=30, green_lower_thresh=75, blue_lower_thresh=60
        )
        & filter_green(
            rgb, red_upper_thresh=195, green_lower_thresh=220, blue_lower_thresh=210
        )
        & filter_green(
            rgb, red_upper_thresh=225, green_lower_thresh=230, blue_lower_thresh=225
        )
        & filter_green(
            rgb, red_upper_thresh=170, green_lower_thresh=210, blue_lower_thresh=200
        )
        & filter_green(
            rgb, red_upper_thresh=20, green_lower_thresh=30, blue_lower_thresh=20
        )
        & filter_green(
            rgb, red_upper_thresh=50, green_lower_thresh=60, blue_lower_thresh=40
        )
        & filter_green(
            rgb, red_upper_thresh=30, green_lower_thresh=50, blue_lower_thresh=35
        )
        & filter_green(
            rgb, red_upper_thresh=65, green_lower_thresh=70, blue_lower_thresh=60
        )
        & filter_green(
            rgb, red_upper_thresh=100, green_lower_thresh=110, blue_lower_thresh=105
        )
        & filter_green(
            rgb, red_upper_thresh=165, green_lower_thresh=180, blue_lower_thresh=180
        )
        & filter_green(
            rgb, red_upper_thresh=140, green_lower_thresh=140, blue_lower_thresh=150
        )
        & filter_green(
            rgb, red_upper_thresh=185, green_lower_thresh=195, blue_lower_thresh=195
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    util.np_info(result, "Filter Green Pen", t.elapsed())
    return result


def filter_blue(
    rgb,
    red_upper_thresh,
    green_upper_thresh,
    blue_lower_thresh,
    output_type="bool",
    display_np_info=False,
):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.

    Args:
      rgb: RGB image as a NumPy array.
      red_upper_thresh: Red channel upper threshold value.
      green_upper_thresh: Green channel upper threshold value.
      blue_lower_thresh: Blue channel lower threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      display_np_info: If True, display NumPy array info and filter time.

    Returns:
      NumPy array representing the mask.
    """
    if display_np_info:
        t = Time()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if display_np_info:
        util.np_info(result, "Filter Blue", t.elapsed())
    return result


def filter_blue_pen(rgb, output_type="bool"):
    """
    Create a mask to filter out blue pen marks from a slide.

    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array representing the mask.
    """
    t = Time()
    result = (
        filter_blue(
            rgb, red_upper_thresh=60, green_upper_thresh=120, blue_lower_thresh=190
        )
        & filter_blue(
            rgb, red_upper_thresh=120, green_upper_thresh=170, blue_lower_thresh=200
        )
        & filter_blue(
            rgb, red_upper_thresh=175, green_upper_thresh=210, blue_lower_thresh=230
        )
        & filter_blue(
            rgb, red_upper_thresh=145, green_upper_thresh=180, blue_lower_thresh=210
        )
        & filter_blue(
            rgb, red_upper_thresh=37, green_upper_thresh=95, blue_lower_thresh=160
        )
        & filter_blue(
            rgb, red_upper_thresh=30, green_upper_thresh=65, blue_lower_thresh=130
        )
        & filter_blue(
            rgb, red_upper_thresh=130, green_upper_thresh=155, blue_lower_thresh=180
        )
        & filter_blue(
            rgb, red_upper_thresh=40, green_upper_thresh=35, blue_lower_thresh=85
        )
        & filter_blue(
            rgb, red_upper_thresh=30, green_upper_thresh=20, blue_lower_thresh=65
        )
        & filter_blue(
            rgb, red_upper_thresh=90, green_upper_thresh=90, blue_lower_thresh=140
        )
        & filter_blue(
            rgb, red_upper_thresh=60, green_upper_thresh=60, blue_lower_thresh=120
        )
        & filter_blue(
            rgb, red_upper_thresh=110, green_upper_thresh=110, blue_lower_thresh=175
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    util.np_info(result, "Filter Blue Pen", t.elapsed())
    return result


def filter_grays(rgb, tolerance=15, output_type="bool"):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.

    Args:
      np_img: RGB image as a NumPy array.
      tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
      output_type: Type of array to return (bool, float, or uint8).

    Returns:
      NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    t = Time()
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    util.np_info(result, "Filter Grays", t.elapsed())
    return result


def apply_image_filters(np_img, slide_num=None, info=None, save=False, display=False):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images.

    Args:
      np_img: Image as NumPy array.
      slide_num: The slide number (used for saving/displaying).
      info: Dictionary of slide information (used for HTML display).
      save: If True, save image.
      display: If True, display image.

    Returns:
      Resulting filtered image as a NumPy array.
    """
    rgb = np_img
    save_display(save, display, info, rgb, slide_num, 1, "Original", "rgb")

    mask_not_green = filter_green_channel(rgb)
    rgb_not_green = util.mask_rgb(rgb, mask_not_green)
    save_display(
        save, display, info, rgb_not_green, slide_num, 2, "Not Green", "rgb-not-green"
    )

    mask_not_gray = filter_grays(rgb)
    rgb_not_gray = util.mask_rgb(rgb, mask_not_gray)
    save_display(
        save, display, info, rgb_not_gray, slide_num, 3, "Not Gray", "rgb-not-gray"
    )

    mask_no_red_pen = filter_red_pen(rgb)
    rgb_no_red_pen = util.mask_rgb(rgb, mask_no_red_pen)
    save_display(
        save,
        display,
        info,
        rgb_no_red_pen,
        slide_num,
        4,
        "No Red Pen",
        "rgb-no-red-pen",
    )

    mask_no_green_pen = filter_green_pen(rgb)
    rgb_no_green_pen = util.mask_rgb(rgb, mask_no_green_pen)
    save_display(
        save,
        display,
        info,
        rgb_no_green_pen,
        slide_num,
        5,
        "No Green Pen",
        "rgb-no-green-pen",
    )

    mask_no_blue_pen = filter_blue_pen(rgb)
    rgb_no_blue_pen = util.mask_rgb(rgb, mask_no_blue_pen)
    save_display(
        save,
        display,
        info,
        rgb_no_blue_pen,
        slide_num,
        6,
        "No Blue Pen",
        "rgb-no-blue-pen",
    )

    mask_gray_green_pens = (
        mask_not_gray
        & mask_not_green
        & mask_no_red_pen
        & mask_no_green_pen
        & mask_no_blue_pen
    )
    rgb_gray_green_pens = util.mask_rgb(rgb, mask_gray_green_pens)
    save_display(
        save,
        display,
        info,
        rgb_gray_green_pens,
        slide_num,
        7,
        "Not Gray, Not Green, No Pens",
        "rgb-no-gray-no-green-no-pens",
    )

    mask_remove_small = filter_remove_small_objects(
        mask_gray_green_pens, min_size=500, output_type="bool"
    )
    rgb_remove_small = util.mask_rgb(rgb, mask_remove_small)
    save_display(
        save,
        display,
        info,
        rgb_remove_small,
        slide_num,
        8,
        "Not Gray, Not Green, No Pens,\nRemove Small Objects",
        "rgb-not-green-not-gray-no-pens-remove-small",
    )

    img = rgb_remove_small
    return img


def apply_filters_to_image_array(
    slide_num, img, slinfo: SlideInfo = None, save=True, display=False
):
    """
    Apply a set of filters to an image and optionally save and/or display filtered images.

    Args:
      slide_num: The slide number.
      save: If True, save filtered images.
      display: If True, display filtered images to screen.

    Returns:
      Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
      (used for HTML page generation).
    """
    t = Time()
    print("Processing slide #%d" % slide_num)

    info = dict()

    if save and not os.path.exists(slide.FILTER_DIR):
        os.makedirs(slide.FILTER_DIR)

    np_orig = img
    filtered_np_img = apply_image_filters(
        np_orig, slide_num, info, save=save, display=display
    )

    if save:
        t1 = Time()
        result_path = slide.get_filter_image_result(slide_num, slinfo=slinfo)
        pil_img = util.np_to_pil(filtered_np_img)
        pil_img.save(result_path)
        print(
            "%-20s | Time: %-14s  Name: %s"
            % ("Save Image", str(t1.elapsed()), result_path)
        )

        t1 = Time()
        thumbnail_path = slide.get_filter_thumbnail_result(slide_num, slinfo=slinfo)
        slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_path)
        print(
            "%-20s | Time: %-14s  Name: %s"
            % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path)
        )

    print("Slide #%03d processing time: %s\n" % (slide_num, str(t.elapsed())))

    return filtered_np_img, info


def save_display(
    save,
    display,
    info,
    np_img,
    slide_num,
    filter_num,
    display_text,
    file_text,
    display_mask_percentage=True,
):
    """
    Optionally save an image and/or display the image.

    Args:
      save: If True, save filtered images.
      display: If True, display filtered images to screen.
      info: Dictionary to store filter information.
      np_img: Image as a NumPy array.
      slide_num: The slide number.
      filter_num: The filter number.
      display_text: Filter display name.
      file_text: Filter name for file.
      display_mask_percentage: If True, display mask percentage on displayed slide.
    """
    mask_percentage = None
    if display_mask_percentage:
        mask_percentage = mask_percent(np_img)
        display_text = (
            display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
        )
    if slide_num is None and filter_num is None:
        pass
    elif filter_num is None:
        display_text = "S%03d " % slide_num + display_text
    elif slide_num is None:
        display_text = "F%03d " % filter_num + display_text
    else:
        display_text = "S%03d-F%03d " % (slide_num, filter_num) + display_text
    if display:
        util.display_img(np_img, display_text)
    if save:
        save_filtered_image(np_img, slide_num, filter_num, file_text)
    if info is not None:
        info[slide_num * 1000 + filter_num] = (
            slide_num,
            filter_num,
            display_text,
            file_text,
            mask_percentage,
        )


def mask_percentage_text(mask_percentage):
    """
    Generate a formatted string representing the percentage that an image is masked.

    Args:
      mask_percentage: The mask percentage.

    Returns:
      The mask percentage formatted as a string.
    """
    return "%3.2f%%" % mask_percentage


def save_filtered_image(np_img, slide_num, filter_num, filter_text):
    """
    Save a filtered image to the file system.

    Args:
      np_img: Image as a NumPy array.
      slide_num:  The slide number.
      filter_num: The filter number.
      filter_text: Descriptive text to add to the image filename.
    """
    t = Time()
    filepath = slide.get_filter_image_path(slide_num, filter_num, filter_text)
    pil_img = util.np_to_pil(np_img)
    pil_img.save(filepath)
    print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath))

    t1 = Time()
    thumbnail_filepath = slide.get_filter_thumbnail_path(
        slide_num, filter_num, filter_text
    )
    slide.save_thumbnail(pil_img, slide.THUMBNAIL_SIZE, thumbnail_filepath)
    print(
        "%-20s | Time: %-14s  Name: %s"
        % ("Save Thumbnail", str(t1.elapsed()), thumbnail_filepath)
    )
