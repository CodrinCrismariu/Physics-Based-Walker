from dataclasses import dataclass, field
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence, float32
import cyclonedds.idl.annotations as annotate

WIDTH = 32
HEIGHT = 24
PIXELS = WIDTH * HEIGHT

@dataclass
@annotate.final
@annotate.autoid("sequential")
class DepthImage_(IdlStruct, typename="DepthImage_"):
    timestamp_us: int = 0
    width       : int = WIDTH
    height      : int = HEIGHT
    depth_data  : sequence[float32] = field(
        default_factory=lambda: [0.0] * PIXELS    
    )