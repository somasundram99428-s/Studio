"""Deluxe Digital Studio - Android app (Kivy), black & white monochrome UI."""
import os, io, threading, traceback, datetime
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle, Line
from kivy.graphics.texture import Texture
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image as KImage
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen, NoTransition
from kivy.uix.scrollview import ScrollView
from kivy.utils import platform
from PIL import Image, ImageOps

import core

BLACK, WHITE, GREY, LIGHT = (0, 0, 0, 1), (1, 1, 1, 1), (0.35, 0.35, 0.35, 1), (0.92, 0.92, 0.92, 1)
Window.clearcolor = WHITE
IS_ANDROID = platform == "android"
LAYOUTS = [("4 COPIES", 4), ("8 COPIES", 8), ("FULL PHOTO 6x4", 0)]


def pil_to_texture(im):
    im = im.convert("RGB")
    tex = Texture.create(size=im.size, colorfmt="rgb")
    tex.blit_buffer(im.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    tex.flip_vertical()
    return tex


def scan_photos(limit=300):
    roots = []
    if IS_ANDROID:
        base = "/storage/emulated/0"
        roots = [f"{base}/DCIM", f"{base}/Pictures", f"{base}/Download"]
    else:
        roots = [os.path.expanduser("~/Pictures")]
    found = []

    def walk(root, depth=0):
        if depth > 4:
            return
        try:
            with os.scandir(root) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if not e.name.startswith("."):
                                walk(e.path, depth + 1)
                        elif e.name.lower().endswith((".jpg", ".jpeg", ".png")):
                            found.append((e.stat().st_mtime, e.path))
                    except OSError:
                        pass
        except OSError:
            pass

    for r in roots:
        walk(r)
    found.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in found[:limit]]


def save_to_gallery(pil_img, name):
    buf = io.BytesIO()
    pil_img.save(buf, "JPEG", quality=95, dpi=(core.DPI, core.DPI), subsampling=0)
    data = buf.getvalue()
    if not IS_ANDROID:
        d = os.path.expanduser("~/Pictures/StudioPrints")
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, name)
        open(p, "wb").write(data)
        return p, False
    from jnius import autoclass
    act = autoclass("org.kivy.android.PythonActivity").mActivity
    sdk = autoclass("android.os.Build$VERSION").SDK_INT
    if sdk >= 29:
        ContentValues = autoclass("android.content.ContentValues")
        Media = autoclass("android.provider.MediaStore$Images$Media")
        cv = ContentValues()
        cv.put("_display_name", name)
        cv.put("mime_type", "image/jpeg")
        cv.put("relative_path", "Pictures/StudioPrints")
        res = act.getContentResolver()
        uri = res.insert(Media.EXTERNAL_CONTENT_URI, cv)
        out = res.openOutputStream(uri)
        out.write(data)
        out.flush()
        out.close()
        return uri, True
    d = "/storage/emulated/0/Pictures/StudioPrints"
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name)
    open(p, "wb").write(data)
    autoclass("android.media.MediaScannerConnection").scanFile(act, [p], None, None)
    return p, False


def share_image(ref, is_uri):
    if not IS_ANDROID:
        return
    from jnius import autoclass, cast
    Intent = autoclass("android.content.Intent")
    String = autoclass("java.lang.String")
    act = autoclass("org.kivy.android.PythonActivity").mActivity
    uri = ref
    if not is_uri:
        Uri = autoclass("android.net.Uri")
        File = autoclass("java.io.File")
        uri = Uri.fromFile(File(ref))
    it = Intent(Intent.ACTION_SEND)
    it.setType("image/jpeg")
    it.putExtra(Intent.EXTRA_STREAM, cast("android.os.Parcelable", uri))
    it.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    act.startActivity(Intent.createChooser(it, cast("java.lang.CharSequence", String("Print / Share"))))


class MonoButton(Button):
    def __init__(self, text, inverted=False, **kw):
        super().__init__(text=text, background_normal="", background_down="", background_color=(0, 0, 0, 0),
                         bold=True, font_size=sp(17), size_hint_y=None, height=dp(54), **kw)
        self.inverted = inverted
        self._off = False
        self.bind(pos=self._draw, size=self._draw, state=self._draw)
        self._draw()

    def set_off(self, off):
        self._off = off
        self.disabled = off
        self._draw()

    def _draw(self, *a):
        fill = BLACK if self.inverted else WHITE
        txt = WHITE if self.inverted else BLACK
        if self.state == "down":
            fill, txt = (GREY, WHITE) if self.inverted else (LIGHT, BLACK)
        if self._off:
            fill, txt = (LIGHT, GREY)
        self.color = txt
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*fill)
            Rectangle(pos=self.pos, size=self.size)
            Color(*BLACK)
            Line(rectangle=(self.x, self.y, self.width, self.height), width=dp(1.6))


class Bar(Label):
    def __init__(self, text, **kw):
        super().__init__(text=text, color=WHITE, bold=True, font_size=sp(20), size_hint_y=None, height=dp(56), **kw)
        with self.canvas.before:
            Color(*BLACK)
            self.r = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=lambda *_: setattr(self.r, "pos", self.pos), size=lambda *_: setattr(self.r, "size", self.size))


class Thumb(Button):
    def __init__(self, path, on_pick, **kw):
        super().__init__(background_normal="", background_down="", background_color=LIGHT, **kw)
        self.path = path
        self.bind(on_release=lambda *_: on_pick(path))
        self.img = KImage(allow_stretch=True, keep_ratio=False)
        self.add_widget(self.img)
        self.bind(pos=self._lay, size=self._lay)

    def _lay(self, *a):
        self.img.pos, self.img.size = self.pos, self.size

    def set_tex(self, tex):
        self.img.texture = tex


class Gallery(Screen):
    def __init__(self, app, **kw):
        super().__init__(name="gallery", **kw)
        self.app = app
        root = BoxLayout(orientation="vertical")
        root.add_widget(Bar("DELUXE DIGITAL STUDIO"))
        row = BoxLayout(size_hint_y=None, height=dp(60), padding=dp(6), spacing=dp(6))
        b1 = MonoButton("REFRESH", inverted=True)
        b1.bind(on_release=lambda *_: self.load())
        b2 = MonoButton("BROWSE FILES")
        b2.bind(on_release=lambda *_: self.browse())
        row.add_widget(b1)
        row.add_widget(b2)
        root.add_widget(row)
        self.hint = Label(text="TAP A PHOTO", color=BLACK, bold=True, size_hint_y=None, height=dp(30))
        root.add_widget(self.hint)
        sv = ScrollView(bar_color=BLACK, bar_width=dp(4))
        self.grid = GridLayout(cols=3, spacing=dp(3), padding=dp(3), size_hint_y=None)
        self.grid.bind(minimum_height=self.grid.setter("height"))
        sv.add_widget(self.grid)
        root.add_widget(sv)
        self.add_widget(root)

    def load(self):
        self.hint.text = "SEARCHING PHOTOS..."
        threading.Thread(target=self._scan, daemon=True).start()

    def _scan(self):
        files = scan_photos()
        Clock.schedule_once(lambda dt: self._fill(files))

    def _fill(self, files):
        self.grid.clear_widgets()
        if not files:
            self.hint.text = "NO PHOTOS FOUND - USE BROWSE FILES"
            return
        self.hint.text = f"TAP A PHOTO  ({len(files)})"
        side = Window.width / 3 - dp(4)
        self.thumbs = []
        for p in files:
            t = Thumb(p, self.app.select, size_hint_y=None, height=side)
            self.grid.add_widget(t)
            self.thumbs.append(t)
        threading.Thread(target=self._thumbs, args=(list(self.thumbs),), daemon=True).start()

    def _thumbs(self, thumbs):
        for t in thumbs:
            try:
                im = Image.open(t.path)
                im.draft("RGB", (300, 300))
                im = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (240, 240), Image.BILINEAR)
                data = im.tobytes()
            except Exception:
                continue
            Clock.schedule_once(lambda dt, t=t, d=data: t.set_tex(self._tex(d)))

    @staticmethod
    def _tex(data):
        tex = Texture.create(size=(240, 240), colorfmt="rgb")
        tex.blit_buffer(data, colorfmt="rgb", bufferfmt="ubyte")
        tex.flip_vertical()
        return tex

    def browse(self):
        try:
            from plyer import filechooser
            filechooser.open_file(on_selection=lambda sel: sel and Clock.schedule_once(lambda dt: self.app.select(sel[0])),
                                  filters=[["Images", "*.jpg", "*.jpeg", "*.png"]])
        except Exception:
            self.hint.text = "FILE BROWSER NOT AVAILABLE"


class Studio(Screen):
    def __init__(self, app, **kw):
        super().__init__(name="studio", **kw)
        self.app = app
        root = BoxLayout(orientation="vertical")
        root.add_widget(Bar("CHECK THE PREVIEW"))
        frame = BoxLayout(padding=dp(8))
        self.prev = KImage(allow_stretch=True, keep_ratio=True)
        frame.add_widget(self.prev)
        root.add_widget(frame)
        self.status = Label(text="", color=BLACK, bold=True, size_hint_y=None, height=dp(64), halign="center",
                            valign="middle", font_size=sp(14))
        self.status.bind(size=lambda *_: setattr(self.status, "text_size", (self.status.width - dp(12), None)))
        root.add_widget(self.status)
        lay = BoxLayout(size_hint_y=None, height=dp(54), spacing=dp(6), padding=(dp(6), 0))
        self.lay_btns = []
        for text, n in LAYOUTS:
            b = MonoButton(text)
            b.font_size = sp(14)
            b.bind(on_release=lambda _b, n=n: app.pick_layout(n))
            lay.add_widget(b)
            self.lay_btns.append(b)
        root.add_widget(lay)
        pad = BoxLayout(orientation="vertical", size_hint_y=None, height=dp(124), spacing=dp(6), padding=dp(6))
        self.save_btn = MonoButton("SAVE & PRINT / SHARE", inverted=True)
        self.save_btn.bind(on_release=lambda *_: app.save_and_share())
        back = MonoButton("<  BACK TO PHOTOS")
        back.bind(on_release=lambda *_: app.go_back())
        pad.add_widget(self.save_btn)
        pad.add_widget(back)
        root.add_widget(pad)
        self.add_widget(root)
        self.set_busy(True)

    def set_busy(self, busy, can_save=False):
        for b in self.lay_btns:
            b.set_off(busy)
        self.save_btn.set_off(not can_save)


class StudioApp(App):
    title = "Deluxe Digital Studio"

    def build(self):
        self.sel = self.full = self.loc = self.sheet = None
        self.hi, self.busy, self.hi_lock = None, False, threading.Lock()
        self.sm = ScreenManager(transition=NoTransition())
        self.gallery, self.studio = Gallery(self), Studio(self)
        self.sm.add_widget(self.gallery)
        self.sm.add_widget(self.studio)
        Window.bind(on_keyboard=self._key)
        if IS_ANDROID:
            from android.permissions import request_permissions, Permission
            request_permissions([Permission.READ_EXTERNAL_STORAGE, Permission.WRITE_EXTERNAL_STORAGE,
                                 "android.permission.READ_MEDIA_IMAGES"],
                                lambda perms, res: Clock.schedule_once(lambda dt: self.gallery.load(), 0.3))
        else:
            Clock.schedule_once(lambda dt: self.gallery.load(), 0.3)
        return self.sm

    def _key(self, win, key, *a):
        if key == 27 and self.sm.current == "studio":
            self.go_back()
            return True
        return False

    def say(self, text):
        self.studio.status.text = text

    def go_back(self):
        if self.hi:
            self.hi["cancel"] = True
        self.sm.current = "gallery"

    def show(self, im):
        self.studio.prev.texture = pil_to_texture(im)

    def select(self, path):
        if self.busy:
            return
        self.busy = True
        self.sel, self.sheet = path, None
        if self.hi:
            self.hi["cancel"] = True
        self.sm.current = "studio"
        self.studio.prev.texture = None
        self.studio.set_busy(True)
        self.say("Checking photo, please wait...")
        pw, ph = round(core.PH_W * core.PREVIEW_SCALE), round(core.PH_H * core.PREVIEW_SCALE)

        def work():
            try:
                full = core.open_photo(path)
                if max(full.size) > 3600:
                    full.thumbnail((3600, 3600), Image.LANCZOS)
                err = loc = None
                try:
                    loc = core.locate_face(full)
                    pv, found, low, bg_ok = core.passport_crop(full, loc, core.PREVIEW_SCALE)
                except Exception as e:
                    err = str(e)
                    pv = core.enhance(ImageOps.fit(full, (pw, ph), Image.BICUBIC, centering=(0.5, 0.4)), core.PREVIEW_SCALE)
                    found = low = bg_ok = False
            except Exception as e:
                Clock.schedule_once(lambda dt: self._failed(f"Cannot open this photo: {e}"))
                return
            Clock.schedule_once(lambda dt: self._selected(path, full, loc, pv, found, low, bg_ok, err))

        threading.Thread(target=work, daemon=True).start()

    def _failed(self, msg):
        self.busy = False
        self.say(msg)

    def _selected(self, path, full, loc, pv, found, low, bg_ok, err):
        if self.sel != path:
            return
        self.full, self.loc, self.busy = full, loc, False
        self.show(pv)
        if err:
            self.say("Could not process this photo. Centred crop shown")
        elif not found:
            self.say("No face found - centred crop. Tap a layout")
        elif low:
            self.say("Face aligned, but face is small - print may be soft")
        elif core.CHANGE_BG and not bg_ok:
            self.say("Face aligned; background kept (could not replace cleanly)")
        else:
            self.say("Face aligned, blue background. Now tap a layout")
        self.studio.set_busy(False)
        job = {"cancel": False, "done": threading.Event(), "img": None}
        self.hi = job

        def hi():
            with self.hi_lock:
                if not job["cancel"]:
                    try:
                        job["img"] = core.passport_crop(full, loc)[0]
                    except Exception:
                        traceback.print_exc()
            job["done"].set()

        threading.Thread(target=hi, daemon=True).start()

    def pick_layout(self, copies):
        if self.busy or self.full is None:
            return
        self.busy = True
        self.studio.set_busy(True)
        self.say("Building the sheet at full quality, please wait...")
        full, loc, job = self.full, self.loc, self.hi

        def work():
            try:
                if copies == 0:
                    sheet = core.make_full(full)
                else:
                    passport = None
                    if job is not None:
                        job["done"].wait()
                        passport = job["img"]
                    if passport is None:
                        passport = core.passport_crop(full, loc)[0]
                    sheet = core.make_sheet(passport, copies)
                pv = ImageOps.expand(sheet.resize((900, round(900 * sheet.height / sheet.width)), Image.LANCZOS),
                                     border=2, fill=(120, 120, 120))
            except Exception as e:
                Clock.schedule_once(lambda dt: self._layout_failed(str(e)))
                return
            Clock.schedule_once(lambda dt: self._layout_done(copies, sheet, pv))

        threading.Thread(target=work, daemon=True).start()

    def _layout_failed(self, msg):
        self.busy = False
        self.say("Could not build layout: " + msg)
        self.studio.set_busy(False)

    def _layout_done(self, copies, sheet, pv):
        self.busy, self.sheet = False, sheet
        self.show(pv)
        self.say("Sheet ready. Check it, then tap SAVE & PRINT / SHARE")
        self.studio.set_busy(False, can_save=True)

    def save_and_share(self):
        if self.busy or self.sheet is None:
            return
        name = "Studio_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
        try:
            ref, is_uri = save_to_gallery(self.sheet, name)
            self.say("Saved to Gallery > Pictures/StudioPrints. Choose Print or Share")
            share_image(ref, is_uri)
        except Exception as e:
            traceback.print_exc()
            self.say("Could not save: " + str(e)[:80])


if __name__ == "__main__":
    StudioApp().run()
