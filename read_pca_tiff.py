from src.hsi_moss.moss2 import *
from matplotlib import pyplot as plt

stiff = STiff(
    Path(
        r"I:\moss_data\Austin moss 2023\Moss\pipeline\03a-decorrelation\t1s01A.pca.tif"
    ),
    TiffOptions(),
)
stiff.type = STIFF_TYPE.PCA
stiff.render8bit()
plt.imshow(stiff.rgb)
plt.show()
print("done")
