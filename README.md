# Soft nudge

![Example](https://github.com/80sVectorz/soft_nudge/blob/master/images/Example.gif?raw=true)

Soft nudge is a module with the goal to provide a reminder without the invasive beeps of conventional timers. I made it to solve a personal issue, but the code is free to use under the GPL-3.0 license.

# Original problem & usage example

Soft nudge was made to serve as a non-invasive reminder that would trigger every 20 minutes. In my case to remind me to look outside and give my eyes a break.

# Basic usage
You can find a simple overview of how to use this package [here](https://github.com/80sVectorz/soft_nudge/blob/master/BasicUsage.md).


# Disclaimer
**Soft nudge only supports windows.**
**The current version of Soft nudge requires a GPU and Cuda using Numba Cuda**

**The animation does not overlay on full screen programs that claim the screen and don't use the windowing system E.G. full-screen minecraft.**

As this project was made for personal purposes first, it's probably not the cleanest code base.
Pull requests are welcome if you're willing to spend the time. If anyone wants to write a non GPU method to show the animation feel free to do so.

# Dependencies
* wxPython
* pywin32
* CTypes
* Numba Cuda
* time

## Animation graphs (Desmos)
* Wave animation: https://www.desmos.com/calculator/gnj2fjlpll
* Pop in time curve: https://www.desmos.com/calculator/orf5s78po2