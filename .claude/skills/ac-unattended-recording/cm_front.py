"""Move any Content Manager result dialog clear of the Claude panel and focus CM's main window."""
import ctypes
from ctypes import wintypes
from capture.autostop import window_exe
u = ctypes.windll.user32
found = []
@ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
def cb(h, l):
    if u.IsWindowVisible(h) and window_exe(h) == "content manager.exe":
        t = ctypes.create_unicode_buffer(100); u.GetWindowTextW(h, t, 100); r = wintypes.RECT(); u.GetWindowRect(h, ctypes.byref(r))
        found.append((h, t.value, (r.left, r.top, r.right, r.bottom)))
    return True
u.EnumWindows(cb, 0)
for h, t, r in found:
    if not t.startswith("Content Manager"):
        print(t, "dialog moved:", bool(u.MoveWindow(h, 300, 380, r[2] - r[0], r[3] - r[1], True)))
main = [h for h, t, r in found if t.startswith("Content Manager")]
if main: print("CM focus:", bool(u.SetForegroundWindow(main[0])))
