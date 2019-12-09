# %%

import time
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

from face.pose import Face

# %%

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facedetector = Face(device=DEVICE)
img = Image.open("test.jpg")

t0 = time.time()
heatmaps = facedetector._detect(img)
lmk = facedetector._compute_peaks_from_heatmaps(heatmaps[:-1])
t1 = time.time()

print(f"time: {t1-t0:0.1f}, {len(lmk)} landmarks")
lmk = np.array([[lm[0], lm[1]] for lm in lmk if lm is not None])

# %%

fig, ax = plt.subplots(1,2, figsize=[9, 6], sharey=True)
ax[0].imshow(img)
ax[0].scatter(lmk[:, 0], lmk[:, 1], c="r")
ax[1].imshow(heatmaps[:68].sum(0))
ax[0].axis("off")
ax[1].axis("off")
plt.tight_layout()
fig.savefig("img/heatmap.jpg")
