# from src.hsi_moss.moss import Moss
from src.hsi_moss.moss2 import *

# f = requests.get('http://localhost:8000/stiff_original/t1s01A.tif', stream=True)
# p = io.BytesIO(f.content)
# cube, wavelengths, rgb, metadata = read_stiff(p)
# plt.imshow(rgb)

# print('done')
# pipeline = MossProcessor(
#     r"I:\moss_data\Austin moss 2023\Moss\pipeline",
#     r"I:\moss_data\Austin moss 2023\Moss\pipeline",
# )
basepath = Path(r"I:\moss_data\Austin moss 2023\Moss\pipeline")
pipeline = MossProcessor(basepath, basepath)

pipeline.process(
    steps=[
        MossProcessor.STEPS.decorrelation.name,
    ]
)
# pipeline.process(
#     range(2),
#     steps=[
#         "segment_kmeans",
#     ],
#     overwrite=True,
# )

print("done")
