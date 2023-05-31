import ctypes
from ctypes import c_bool, c_byte, c_int, c_float, c_ubyte
from ctypes.wintypes import COLORREF, DWORD, HBITMAP, HRGN, HWND, HDC, POINT, SIZE
import win32con
import win32gui
import wx
import time
import soft_nudge_cuda


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp", c_ubyte),
        ("BlendFlags", c_ubyte),
        ("SourceConstantAlpha", c_ubyte),
        ("AlphaFormat", c_ubyte),
    ]


class Frame(wx.Frame):
    def __init__(
        self,
        parent=None,
        color=(36, 173, 243, 20),
        period=14,
        amplitude=0.02,
        duration=10.0,
        trend_split=0.6,
        flat_time_pct=0.4,
    ):
        wx.Frame.__init__(
            self,
            parent,
            size=(500, 500),
            style=wx.STAY_ON_TOP | wx.CLIP_CHILDREN | wx.TRANSPARENT_WINDOW,
        )

        hwnd = self.GetHandle()

        extended_style_settings = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(
            hwnd,
            win32con.GWL_EXSTYLE,
            extended_style_settings
            | win32con.WS_EX_LAYERED
            | win32con.WS_POPUP
            | win32con.WS_VISIBLE
            | win32con.WS_EX_TRANSPARENT
            | win32con.WS_EX_TOPMOST,
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

    def on_timer(self, event):
        self.Refresh(True)

    def on_size(self, event):
        event.Skip()
        self.Refresh()

    def layered_update(self, dc, blend_func):
        # Code has been translated/inferred using: https://www.vbforums.com/showthread.php?888761-UpdateLayeredWindow()-Drove-Me-Crazy
        # https://stackoverflow.com/questions/43712796/draw-semitransparently-in-invisible-layered-window
        rect = self.GetRect()
        w, h = self.GetSize()
        px, py = rect.GetPosition()
        hwnd = self.GetHandle()

        res = ctypes.windll.user32.UpdateLayeredWindow(
            HWND(hwnd),
            HDC(wx.ScreenDC().GetHandle()),
            ctypes.pointer(POINT(px, py)),
            ctypes.pointer(SIZE(w, h)),
            HDC(dc.GetHandle()),
            ctypes.pointer(POINT(0, 0)),
            COLORREF(0),
            ctypes.pointer(blend_func),
            DWORD(win32con.ULW_ALPHA),
        )
        if res == 0:
            print(ctypes.windll.kernel32.GetLastError())

    def on_paint(self, event):
        w, h = self.GetSize()

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
            self.Close(force=True)
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
    color_rgba,
    anim_period,
    anim_amplitude,
    duration=10.0,
    trend_split=0.6,
    flat_time_pct=0.4,
):
    app = wx.App(False)
    frame = Frame(
        color=color_rgba,
        period=anim_period,
        amplitude=anim_amplitude,
        duration=duration,
        trend_split=trend_split,
        flat_time_pct=flat_time_pct,
    )
    frame.ShowFullScreen(True)
    app.MainLoop()


if __name__ == "__main__":
    main()
