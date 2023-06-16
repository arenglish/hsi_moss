# from src.hsi_moss.moss import Moss
from src.hsi_moss.moss2 import *

pipeline = MossProcessor(
    r"I:\moss_data\Austin moss 2023\Moss\pipeline",
    r"I:\moss_data\Austin moss 2023\Moss\pipeline",
)

# pipeline.process([77, 173, 269, 365])
pipeline.process(steps=["correction_grayref", "correction_whiteref"])
