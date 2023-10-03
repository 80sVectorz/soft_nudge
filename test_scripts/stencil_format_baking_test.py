import soft_nudge
from soft_nudge import baking
from soft_nudge.baking import BakedAnimationFormat
from numba import cuda
import time

print("Output file: ./baked_animations_cache/stencil_format_test.sna")
time.sleep(3)
print("Baking may take a while depending on hardware")
t0 = time.time_ns()
animation = baking.bake_animation(
    color_rgba=(30, 173, 243, 40), size=(1920,1080),
    fps=20,duration=10.0,
    anim_period=14, anim_amplitude=0.02,anim_undulation_frequency=0.25,compression_method=BakedAnimationFormat.stencil_format,force_cpu=True
)

print(f"Finished baking and compressing! Total time it took was: {(time.time_ns()-t0)/1_000_000_000} seconds.\n Now writing the result to file")
with open("./baked_animations_cache/stencil_format_test.sna","wb") as f:
    animation.write_to_file(f)

print("Loading file and displaying result")
with open("./baked_animations_cache/stencil_format_test.sna","rb") as f:
    soft_nudge.baked_nudge_from_file(f)