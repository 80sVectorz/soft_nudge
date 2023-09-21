import ctypes
from ctypes import c_bool, c_byte, c_int, c_float, c_ubyte
from ctypes.wintypes import COLORREF, DWORD, HBITMAP, HRGN, HWND, HDC, POINT, SIZE
import win32con
import win32gui
import wx
import time
from soft_nudge import soft_nudge_cuda


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
        color: tuple[int,int,int,int] = (36, 173, 243, 20),
        period: float = 14.0,
        amplitude: float = 0.02,
        duration: float = 10.0,
        trend_split: float = 0.6,
        flat_time_pct: float = 0.4,
        size: tuple[int,int] = (500, 500),
        target_display: int = 0,
    ):
        wx.Frame.__init__(
            self,
            parent,
            size=size,
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
        cdata, adata = soft_nudge_cuda.get_bmp_data(
            w,
            h,
            self.anim_color,
            self.anim_period,
            self.anim_amplitude,
            self.duration,
            self.trend_split,
            self.flat_time_pct,
            self.time,
        )
        if cdata[0, 0].tolist() == [101, 110, 100]:
            exit()

        img = wx.Image(w, h)
        img.SetData(cdata)
        img.SetAlpha(adata)
        bmp = img.ConvertToBitmap()
        memdc = wx.MemoryDC(bmp)
        blend_func = BLENDFUNCTION(win32con.AC_SRC_OVER, 0, 255, win32con.AC_SRC_ALPHA)
        self.layered_update(memdc, blend_func)


def main():
    nudge((30, 173, 243, 40), 14, 0.02, duration=6.0)


def nudge(
    color_rgba: tuple[int,int,int,int],
    anim_period: float,
    anim_amplitude: float,
    duration: float = 10.0,
    trend_split: float = 0.6,
    flat_time_pct: float = 0.4,
    target_display: int = 0,
):
    """Starts a soft nudge animation by creating a full screen frame and rendering with the GPU.

    **IMPORTANT**
    This method will kill the current thread when the animation is finished.
    """
    app = wx.App()
    frame = SoftNudgeFrame(
        size=wx.Display(target_display).GetGeometry().GetSize(),
        color=color_rgba,
        period=anim_period,
        amplitude=anim_amplitude,
        duration=duration,
        trend_split=trend_split,
        flat_time_pct=flat_time_pct,
        target_display=target_display,
    )
    frame.Disable()
    frame.Show(True)  # Size is later set to be full screen in the layered update.
    app.MainLoop()


if __name__ == "__main__":
    main()
