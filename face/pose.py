""" Pytorch openpose"""
import logging
import os
import shutil
import tempfile
from urllib.request import urlopen
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from .models import FaceNet


LOG = logging.getLogger(__name__)
FACE_WEIGHTS = os.path.expanduser("~/.dg/openpose/data/facenet.pth")
DEVICE = torch.device("cpu")
TOTEN = ToTensor()
TOPIL = ToPILImage()


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------


def download_url_to_file(url, dst):
    """Download object at the given URL to a local path.

    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. `/path/file`

    """

    u = urlopen(url)

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.

    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)

    with tempfile.NamedTemporaryFile(delete=False, dir=dst_dir) as f:
        name = f.name
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)

    shutil.move(name, dst)
    if os.path.exists(name):
        os.remove(name)


def download_face_weights(dst, force=False):
    """
    Download the weights file to: '~/.dg/openpose/data/facenet.pth'
    """
    if os.path.exists(dst) and not force:
        print(f"Trying existing data at: {dst}")
        return dst
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    url = "https://www.dropbox.com/s/rgtxht3dalsee73/facenet.pth?dl=1"
    print("Downloading 153mb...")
    download_url_to_file(url, dst)
    print("Finished downloading")
    return dst


def load(model, fname):
    """Load the model weights - downloading if necessary.
    """
    try:
        model.load_state_dict(torch.load(fname))
    except OSError as _:
        print("model data not found - attempting download.")
        fname = download_face_weights(fname)
    try:
        model.load_state_dict(torch.load(fname))
    except OSError as _:
        print("model data not loaded - retrying download.")
        fname = download_face_weights(fname, force=True)
    try:
        model.load_state_dict(torch.load(fname))
    except OSError as _:
        print("model data not loaded - Aborting...")
        return False
    return True


params = {
    'gaussian_sigma': 2.5,
    'inference_img_size': 736,  # 368, 736, 1312
    'heatmap_peak_thresh': 0.1,
    'crop_scale': 1.5,
    'line_indices': [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13],
        [13, 14], [14, 15], [15, 16],
        [17, 18], [18, 19], [19, 20], [20, 21],
        [22, 23], [23, 24], [24, 25], [25, 26],
        [27, 28], [28, 29], [29, 30],
        [31, 32], [32, 33], [33, 34], [34, 35],
        [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
        [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
        [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
        [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
        [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66],
        [66, 67], [67, 60]
    ],
}


# -----------------------------------------------------------------------------
# FACE MODEL
# -----------------------------------------------------------------------------


class Face(object):
    """
    The OpenPose face landmark detector model.

    Args:
        inference_size: set the size of the inference image size, suggested:
            368, 736, 1312, default 736
        gaussian_sigma: blur the heatmaps, default 2.5
        heatmap_peak_thresh: return landmark if over threshold, default 0.1

    """
    def __init__(self, device=DEVICE,
                 inference_size=None,
                 gaussian_sigma=None,
                 heatmap_peak_thresh=None):
        self.device = device
        self.inference_size = inference_size or params["inference_img_size"]
        self.sigma = gaussian_sigma or params['gaussian_sigma']
        self.threshold = heatmap_peak_thresh or params["heatmap_peak_thresh"]
        self.model = FaceNet()
        load(self.model, FACE_WEIGHTS)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("using device:", device)

    def detect(self, face_img):
        """face_img is an RGB PIL image """

        heatmaps = self._detect(face_img)
        keypoints = self._compute_peaks_from_heatmaps(heatmaps[:-1])
        return keypoints

    def _detect(self, face_img):
        """face_img is an RGB PIL image """

        face_img_w, face_img_h = face_img.size
        resized_image = Image.fromarray(
            np.array(face_img)[:, :, ::-1]).resize(
                (self.inference_size, self.inference_size),
                resample=2)

        x_data = TOTEN(resized_image) - 0.5
        x_data = x_data.to(self.device)

        with torch.no_grad():
            hs = self.model(x_data[None, ...])
            heatmaps = F.interpolate(
                hs[-1],
                (face_img_h, face_img_w),
                mode='bilinear', align_corners=True).cpu().numpy()[0]
        return heatmaps

    def _compute_peaks_from_heatmaps(self, heatmaps):
        """blur and threshold each heatmap."""
        keypoints = []
        for channel in heatmaps:
            heatmap = gaussian_filter(channel, sigma=self.sigma)
            max_value = heatmap.max()
            if max_value > self.threshold:
                coords = np.array(
                    np.where(heatmap == max_value)).flatten().tolist()
                # x, y, conf
                keypoints.append([coords[1], coords[0], max_value])
            else:
                keypoints.append(None)
        return keypoints
