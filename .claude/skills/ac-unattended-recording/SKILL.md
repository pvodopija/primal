---
name: ac-unattended-recording
description: Record Assetto Corsa bot sessions for PRIMAL unattended on the Windows capture PC - drive Content Manager, AC and OBS with computer use, start/stop OBS by lap count, save the replay, import, decode, check and trim. Use whenever asked to record sessions, pre-flight a new track, or save/import an AC recording.
---

# Unattended AC recording (Windows capture PC)

Learned the hard way over 2026-09-27/28. Follow it literally; every rule below
cost at least one lost session.

## Hard rules

- **Never press Escape.** The Claude app keeps Escape as its "stop computer use"
  key: pressing it stops *you*, and AC never sees it. Open the replay through the
  rig instead (see *Save the replay*).
- **Never `open_application` for AC or OBS.** It launches a second instance.
  Bring windows forward with `SetForegroundWindow` from a script
  (`capture.autostop.focus_ac()`; for CM use the snippet in *Content Manager*).
- **Nothing heavy while OBS records** (no decode, no video reads, no analysis).
  Decode the previous session *before* starting the next recording.
- **AC buttons need hover.** `mouse_move` onto the button, wait ~1 s, then click.
- **Other windows steal focus mid-session** (the Claude app whenever it shows a
  message, Windows' TextInputHost behind its chat box, NVIDIA's overlay). Keys
  sent by computer use can land there. `capture.autostop` re-focuses AC itself
  before every key; do the same (`focus_ac()`) before any click into AC.
- **The Claude panel is always-on-top at the top right** (about physical x
  1372-1908, y 20-748). Anything under it (e.g. CM's result dialog buttons) can't
  be clicked: move that window left with `MoveWindow` first.
- **OBS hotkeys are F13 (start) / F14 (stop).** Never use Alt/Shift chords:
  Ctrl+Alt+Shift+F9/F10 toggled NVIDIA Instant Replay (icon in frame, GPU load).
- AC is capped at 60 fps (`video.ini` `FPS_CAP_MS=16.6667`); the monitor runs 70 Hz.
- Bot strength can't be varied in Hotlap (`AI_LEVEL` stays 100). Vary the car.
- Hotlap's 30-minute timer does not stop the bot; it drives on for hours and the
  replay keeps everything (7.7 h came to 1.1 GB).

## Per session

Sessions pattern: session 1 MX-5 Cup (`ks_mazda_mx5_cup`) ~12:00 scattered
clouds; session 2 Abarth 500 Assetto Corse 18:30 clear. Mode Hotlap.

1. **Content Manager** (Drive > Quick Drive). Bring it forward and move any
   result dialog clear of the Claude panel:
   `PYTHONPATH=. .venv/Scripts/python.exe .claude/skills/ac-unattended-recording/cm_front.py` (EnumWindows over
   `content manager.exe`, `MoveWindow(dialog, 300, 380, ...)`, `SetForegroundWindow(main)`).
   Close the dialog, then: car tile > filter box > type (`mx5`, `abarth 500`) >
   pick > OK; track tile > filter > type > pick layout tile (check the title by
   zooming: the tile row moves down when the title wraps) > OK; time slider
   (0-24 h across the slider; 12:00 is the middle); weather dropdown. Click **Go!**
2. Wait for `acs.exe` with a window, then ~35-40 s more. Check `cfg/race.ini`
   (`TRACK`, `CONFIG_TRACK`, `MODEL`). `focus_ac()`, screenshot: the session menu
   must be showing (not the loading screen).
3. Hover, then click the steering-wheel icon (about (286, 275) in screenshot
   coordinates). Confirm the barcode is visible at the bottom.
4. `python -m capture.autostop --engage-bot` - sends Ctrl+C with scan codes to a
   focused AC and waits until telemetry shows the car moving. It refuses if the
   timecode log isn't fresh (still in the menu).
5. **New track only - pre-flight:** `autostop --start --laps 2 --max-min 7`, then
   `session import <newest D:\Videos mp4> --preflight`, `overlay_decode calibrate
   ... --roi 458,692,378,28`, `overlay_decode decode`, `preflight check`. Labels
   must pass; "camera stays on the tarmac" at <1% is soft (look at the worst frame).
   First-lap hitches on a new track are asset loading.
6. Record: `autostop --start --laps 10 --max-min <~2x expected>` in the background.
   It stops OBS itself, and also on stall, a parked car, or timeout.
7. **Save the replay:** set `replay_now = 1` in
   `apps/lua/primal_rig/rig.txt` (the rig opens the replay and resets the flag;
   `replay_request.txt` says `opened`). `focus_ac()`, move the mouse to the bottom
   of the AC window, click **SAVE REPLAY**, then **Ok**.
8. Import while AC is still on the session:
   `session import <mp4> --split train|holdout --notes "..."`; copy the newest
   `D:\Documents\Assetto Corsa\replay\*.acreplay` to `<session>/replay.acreplay`.
   Import copies only the telemetry chunks written during the recording, so it
   also works hours later.
9. Close AC: `(Get-Process acs).CloseMainWindow()`; it autosaves to `replay\temp`.
10. In the background: calibrate, decode, `preflight check`, `session trim --last-lap`.
    Wait for it to finish before step 6 of the next session.

## Re-rendering a saved replay (no driving)

Set the rig in `apps/lua/primal_rig/rig.txt` first (`fov_deg`, `look`,
`wander_m`, ...; it is re-read twice a second), and put it back afterwards.

1. CM > Media > replays. Click the search box, type part of the replay's name,
   Return, then **Play** (the first click may only activate the window; click
   again if AC doesn't start).
2. Wait for AC (`acs.exe` with a window, then ~40 s) and `focus_ac()`. If CSP
   asks "Did Assetto Corsa just crash?", answer **No** (never send reports).
3. Move the mouse to the bottom of AC's window for the replay bar. Replays
   often start with the car parked for minutes: the green lap markers on the
   timeline show where driving is; click the timeline there. Move the mouse
   away so the bar (which overlaps the barcode) hides.
4. Check the car moves: `last_row(newest_launch())` speed > 5 km/h and a fresh
   chunk. A frozen `cam_s` means paused or parked.
5. `autostop --start --laps 10 --max-min <minutes of replay left>`. The lap
   counter works in replays. When the replay ends the picture freezes but the
   speed does not drop, so the parked guard won't fire; `trim --last-lap`
   removes that tail.
6. `session import <mp4> --rig --split ...` while AC is still open (it writes
   `camera_vfov_deg` from the telemetry), then close AC.
7. After AC closes, focus usually falls to NVIDIA's overlay, Explorer or
   another app. `cm_front.py` attaches to the foreground window's input thread
   for its `SetForegroundWindow` (no keys sent), which brings CM forward on the
   first try (2026-10-07); don't click Explorer to get there (the user declined
   computer-use access to it). CM minimizes itself while AC runs, so it can't
   be clicked beforehand.
8. Decode all renders in one batch after the last recording, not between them.

Also learned 2026-10-07:

- A replay played to its end stays paused at the end: after jumping back on the
  timeline, press the play button in the bar's middle (hover first) and check
  the speed in `last_row(newest_launch())` before recording.
- CSP hot-reloads `primal_rig.lua` when the installed file changes (a new rig log
  starts), so a rig fix doesn't need AC restarted.
- AC's track x is scaled per side of the AI line (-1 at the left edge, `sides.x`
  metres left of it; +1 at the right edge, `sides.y` right), not half the width
  each way. The rig's `wander_full = 1` mode and `pose.npz`'s `lateral_m` use it.
- The replay's own weather and time are in CM's replay details (Weather, Time):
  time of day can't be changed in a replay (CSP's time functions are for
  offline races only), weather can (`weather` in rig.txt, an `ac.WeatherType`).

## New track, live with the rig (2026-10-08)

Rig `replay_only = 0`, `wander_full = 1`, `weather = -1`; CM Hotlap at noon, weather from
CM. Drive icon, then `autostop --engage-bot` **on its own, checking its exit code** (a
`| tail` once hid a failed engage and recorded a parked car), then `--start --laps 10`.
Then `replay_now = 1`, SAVE REPLAY + Ok (the bar shows with the mouse inside the window's
bottom edge, y≈650 in screenshot coordinates), import, copy the newest `.acreplay` into the
session, set rig `weather` to the opposite type, rewind/play, wait for speed > 20 km/h
before `--start` (the parked guard stops a recording of the parked start), import, close.
Put `replay_only = 1` and `weather = -1` back afterwards.

- Light live pre-flight from the telemetry log (label vs car speed, wraps vs lap ticks,
  side widths) after two laps catches a bad AI line without a separate session.
- Suzuka East: the bot never leaves the pit lane; use the GP layout. The installed
  "Daytona Sandown Park" is a UK kart track. Suzuka's crossover snaps s to the other
  level (pack drops it) and its AI side widths are 0 just before the line (the rig holds
  its offset).
- AC sometimes quits or hangs on launch (dxgi `endScene`, or "Race cancelled"): stop the
  hung process, close CM's dialog, Go/Play again; answer CSP's "Did AC crash?" with No.

## If something goes wrong

- A recording of a parked car / wrong settings: move it to `D:\Videos\rejected\`
  (never delete). Relaunch from CM with the same settings.
- AC hung on close (no window, no disk writes for minutes): `Stop-Process`, and
  say the replay is lost.
- Details and measurements: `docs/capture-log.md` (*The 2026-09-27 recording
  session*) and `README.md` §4 (*Unattended recording*).
