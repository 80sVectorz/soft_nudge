import ctypes
from ctypes import c_ubyte
from ctypes.wintypes import COLORREF, DWORD, HBITMAP, HRGN, HWND, HDC, POINT, SIZE
import win32con
import win32gui
import wx
import time
from numba import cuda
from soft_nudge.utils import calculate_limit_rect
from soft_nudge import soft_nudge_cuda, soft_nudge_cpu
from soft_nudge.baking import BakedAnimation
import numpy as np


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", c_ubyte),
        ("BlendFlags", c_ubyte),
        ("SourceConstantAlpha", c_ubyte),
        ("AlphaFormat", c_ubyte),
    ]


class SoftNudgeFrame(wx.Frame):
    def __init__(
        self,
        parent=None,
        color: tuple[int, int, int, int] = (36, 173, 243, 20),
        period: float = 14.0,
        amplitude: float = 0.02,
        undulation_frequency: float = 0.25,
        duration: float = 10.0,
        trend_split: float = 0.6,
        flat_time_pct: float = 0.4,
        size: tuple[int, int] = (500, 500),
        target_display: int = 0,
        pop_in_outside: float = 1.15,
        pop_in_inside: float = 0.9,
        use_cpu: bool = False,
    ):
        wx.Frame.__init__(
            self,
            parent,
            size=size,
            style=wx.STAY_ON_TOP | wx.TRANSPARENT_WINDOW,
        )

        hwnd = self.GetHandle()

        extended_style_settings = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            extended_style_settings
            | win32con.WS_EX_LAYERED
            | win32con.WS_EX_TRANSPARENT,
        )

        self.SetTitle("Soft Nudge")
        self.Center()
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.timer = wx.Timer(self)
        self.timer.Start(2)
        self.start_time = time.time_ns()
        self.time = time.time_ns()
        self.anim_color = color
        self.anim_period = period
        self.anim_amplitude = amplitude
        self.duration = duration
        self.trend_split = trend_split
        self.flat_time_pct = flat_time_pct
        self.target_display = target_display
        self.undulation_frequency = undulation_frequency
        self.pop_in_outside = pop_in_outside
        self.pop_in_inside = pop_in_inside
        self.use_cpu = use_cpu

        self.limit_rect = calculate_limit_rect(
            size[0],
            size[1],
            self.anim_amplitude,
            self.anim_period,
            self.undulation_frequency,
            self.pop_in_inside,
        )

    def on_timer(self, event):
        event.Skip()
        self.Refresh(True)

    def on_size(self, event):
        event.Skip()
        self.Refresh()

    def layered_update(self, dc, blend_func):
        # Code has been translated/inferred using: https://www.vbforums.com/showthread.php?888761-UpdateLayeredWindow()-Drove-Me-Crazy
        # https://stackoverflow.com/questions/43712796/draw-semitransparently-in-invisible-layered-window
        # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-updatelayeredwindow

        screen_geometry = wx.Display(self.target_display).GetGeometry()
        w, h = screen_geometry.GetSize()
        px, py = screen_geometry.GetPosition()
        scrdc = wx.ScreenDC().GetHandle()
        hwnd = self.GetHandle()
        res = ctypes.windll.user32.UpdateLayeredWindow(
            HWND(hwnd),  # [in]           HWND          hWnd,
            HDC(scrdc),  # [in, optional] HDC           hdcDst,
            ctypes.pointer(POINT(px, py)),  # [in, optional] POINT         *pptDst,
            ctypes.pointer(SIZE(w, h)),  # [in, optional] SIZE          *psize,
            HDC(dc.GetHandle()),  # [in, optional] HDC           hdcSrc,
            ctypes.pointer(POINT(0, 0)),  # [in, optional] POINT         *pptSrc,
            COLORREF(0),  # [in]           COLORREF      crKey,
            ctypes.pointer(blend_func),  # [in, optional] BLENDFUNCTION *pblend,
            DWORD(win32con.ULW_ALPHA),  # [in]           DWORD         dwFlags
        )
        if res == 0:
            print(ctypes.windll.kernel32.GetLastError())

    def on_paint(self, event):
        w, h = wx.Display(self.target_display).GetGeometry().GetSize()

        self.time = time.time_ns() - self.start_time

        render_method = soft_nudge_cuda.get_bmp_data
        if self.use_cpu:
            render_method = soft_nudge_cpu.get_bmp_data

        cdata, adata = render_method(
            w,
            h,
            self.anim_color,
            self.anim_period,
            self.anim_amplitude,
            self.undulation_frequency,
            self.duration,
            self.trend_split,
            self.flat_time_pct,
            self.time,
            self.pop_in_outside,
            self.pop_in_inside,
            *self.limit_rect,
        )

        if cdata[0, 0].tolist() == [101, 110, 100]:
            exit()

        img = wx.Image(
            width=w, height=h, data=cdata.astype(np.uint8), alpha=adata.astype(np.uint8)
        )
        bmp = img.ConvertToBitmap()
        memdc = wx.MemoryDC(bmp)
        blend_func = BLENDFUNCTION(win32con.AC_SRC_OVER, 0, 255, win32con.AC_SRC_ALPHA)
        self.layered_update(memdc, blend_func)


class BakedSoftNudgeFrame(wx.Frame):
    def __init__(
        self,
        baked_animation: BakedAnimation,
        parent=None,
        target_display: int = 0,
    ):
        wx.Frame.__init__(
            self,
            parent,
            size=wx.Display(target_display).GetGeometry().GetSize(),
            style=wx.STAY_ON_TOP | wx.CLIP_CHILDREN | wx.TRANSPARENT_WINDOW,
        )

        hwnd = self.GetHandle()

        extended_style_settings = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            extended_style_settings
            | win32con.WS_EX_LAYERED
            | win32con.WS_EX_TRANSPARENT,
        )

        self.SetTitle("Soft Nudge")
        self.Center()
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.animation = baked_animation
        self.fps = self.animation.fps
        self.timer = wx.Timer(self)
        self.timer.Start(1000 // self.fps)
        self.target_display = target_display
        self.current_frame_index = 0

    def on_timer(self, event):
        event.Skip()

        w, h = wx.Display(self.target_display).GetGeometry().GetSize()

        cdata, adata = self.animation.get_frame(self.current_frame_index)

        img = wx.Image(
            width=w, height=h, data=cdata.astype(np.uint8), alpha=adata.astype(np.uint8)
        )
        bmp = img.ConvertToBitmap()
        memdc = wx.MemoryDC(bmp)
        blend_func = BLENDFUNCTION(win32con.AC_SRC_OVER, 0, 255, win32con.AC_SRC_ALPHA)
        self.layered_update(memdc, blend_func)

        self.current_frame_index += 1
        if self.current_frame_index > self.animation.frame_count - 1:
            exit()

        self.Refresh(True)

    def on_size(self, event):
        event.Skip()
        self.Refresh()

    def layered_update(self, dc, blend_func):
        # Code has been translated/inferred using: https://www.vbforums.com/showthread.php?888761-UpdateLayeredWindow()-Drove-Me-Crazy
        # https://stackoverflow.com/questions/43712796/draw-semitransparently-in-invisible-layered-window
        # https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-updatelayeredwindow

        screen_geometry = wx.Display(self.target_display).GetGeometry()
        w, h = screen_geometry.GetSize()
        px, py = screen_geometry.GetPosition()
        scrdc = wx.ScreenDC().GetHandle()
        hwnd = self.GetHandle()
        res = ctypes.windll.user32.UpdateLayeredWindow(
            HWND(hwnd),  # [in]           HWND          hWnd,
            HDC(scrdc),  # [in, optional] HDC           hdcDst,
            ctypes.pointer(POINT(px, py)),  # [in, optional] POINT         *pptDst,
            ctypes.pointer(SIZE(w, h)),  # [in, optional] SIZE          *psize,
            HDC(dc.GetHandle()),  # [in, optional] HDC           hdcSrc,
            ctypes.pointer(POINT(0, 0)),  # [in, optional] POINT         *pptSrc,
            COLORREF(0),  # [in]           COLORREF      crKey,
            ctypes.pointer(blend_func),  # [in, optional] BLENDFUNCTION *pblend,
            DWORD(win32con.ULW_ALPHA),  # [in]           DWORD         dwFlags
        )
        if res == 0:
            print(ctypes.windll.kernel32.GetLastError())


def main():
    nudge((30, 173, 243, 40), 14, 0.02, duration=6.0)


def nudge(
    color_rgba: tuple[int, int, int, int],
    anim_period: float,
    anim_amplitude: float,
    duration: float = 10.0,
    trend_split: float = 0.6,
    flat_time_pct: float = 0.4,
    target_display: int = 0,
    anim_undulation_frequency: float = 0.25,
    pop_in_outside: float = 1.15,
    pop_in_inside: float = 0.9,
    force_cpu: bool = False,
):
    """Starts a soft nudge animation by creating a full screen frame and rendering the animation in real time.
    The device used depends on the availability of CUDA and the value of `force_cpu`.

    **IMPORTANT**
    This method will kill the current thread when the animation is finished.
    """
    use_cpu = True
    if cuda.is_available() and not force_cpu:
        use_cpu = False

    app = wx.App()
    frame = SoftNudgeFrame(
        size=wx.Display(target_display).GetGeometry().GetSize(),
        color=color_rgba,
        period=anim_period,
        amplitude=anim_amplitude,
        undulation_frequency=anim_undulation_frequency,
        duration=duration,
        trend_split=trend_split,
        flat_time_pct=flat_time_pct,
        target_display=target_display,
        pop_in_outside=pop_in_outside,
        pop_in_inside=pop_in_inside,
        use_cpu=use_cpu,
    )
    frame.Disable()
    frame.Show(True)  # Size is later set to be full screen in the layered update.
    app.MainLoop()


def baked_nudge_from_file(
    binary_baked_animation_file_object,
    target_display: int = 0,
):
    """Starts a soft nudge animation by creating a full screen frame and displaying a pre-baked animation.

    **IMPORTANT**
    This method will kill the current thread when the animation is finished.
    """
    baked_animation = BakedAnimation.load_from_file(binary_baked_animation_file_object)
    app = wx.App()
    frame = BakedSoftNudgeFrame(
        baked_animation,
        target_display=target_display,
    )
    frame.Disable()
    frame.Show(True)  # Size is later set to be full screen in the layered update.
    app.MainLoop()

def baked_nudge_from_object(
    baked_animation:BakedAnimation,
    target_display: int = 0,
):
    """Starts a soft nudge animation by creating a full screen frame and displaying a pre-baked animation.

    **IMPORTANT**
    This method will kill the current thread when the animation is finished.
    """
    app = wx.App()
    frame = BakedSoftNudgeFrame(
        baked_animation,
        target_display=target_display,
    )
    frame.Disable()
    frame.Show(True)  # Size is later set to be full screen in the layered update.
    app.MainLoop()



if __name__ == "__main__":
    main()
