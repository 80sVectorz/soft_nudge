# Installation

Simply install soft nudge using pip: `python -m pip install soft-nudge`

# Starting an animation

Starting an animation using soft nudge is quite simple:
```py
import soft_nudge #Import soft-nudge
soft_nudge.nudge(
    color_rgba=(30, 173, 243, 40),
    anim_period=14, anim_amplitude=0.02,
    duration=10.0,anim_undulation_frequency=0.25,  
    force_cpu=True
)
```
This will show the animation and close the program. 
To stop the program from exiting, use a separate process for the animation:
```py
import soft_nudge
import multiprocessing
import time


def anim_worker_function():
    soft_nudge.nudge(
        color_rgba=(30, 173, 243, 40),
        anim_period=14, anim_amplitude=0.02, duration=10.0 
        anim_undulation_frequency=0.25,
        force_cpu=True
    )


def main(): # Show the animation every 20 minutes
    while True:
        anim_process = multiprocessing.Process(
            target=anim_worker_function, name="SoftNudgeAnim"
        )
        anim_process.start()
        time.sleep(20 * 60)
        anim_process.terminate()


if __name__ == "__main__":
    main()
```

To tweak the look of the animations use the graphs listed in the
[read me](https://github.com/80sVectorz/soft_nudge/blob/master/README.md)
to find the correct parameters for your application.

If you need more example code, please take a look at the [test_scripts directory](https://github.com/80sVectorz/soft_nudge/tree/master/test_scripts)