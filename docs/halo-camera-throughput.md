# Halo camera → NPU throughput

Can Brilliant Labs Halo capture frames at a high rate and embed them on its own
NPU, with nothing streamed over Bluetooth? Researched 2026-10-03 against
`brilliantlabsAR/halo-firmware` `main` @ `c2bfc03` (release 0.8.17), its pinned
Zephyr and HAL forks, and the Halo hardware manual in `brilliantlabsAR/docs`.

## Answer

**Nothing found in the hardware or open firmware caps local capture + NPU
embedding in single digits.** The single-digit numbers people report come from
the stock *photo* path (capture → software JPEG → BLE), which is built for
stills, not for throughput.

- **NPU.** PRIMAL's current trunk compiles 100% onto the Ethos-U55 and is
  estimated at **~3 ms per frame (~300 fps)**. NPU compute is not the
  bottleneck.
- **Sensor bus.** The firmware runs the camera at 24 MHz pixel clock,
  8-bit parallel, full VGA. That bus moves at most **~78 full-VGA frames/s**
  before blanking. **120 fps at full VGA is impossible in the shipped
  configuration.**
- **Reaching 120 fps** needs one of two things, and neither is verified yet:
  - the 48 MHz register table PixArt left commented out in the driver (bus
    ceiling ~156 VGA fps), or
  - a smaller window from the sensor. PRIMAL only needs ~160×120, so this is
    the more natural route.

Two things the open sources don't settle, and only an on-device test will:
the PAG7982's true maximum frame rate, and the B1 LP-CPI's maximum pixel
clock. Their datasheets were not reachable from this environment. **Run the
test in [Decisive test](#decisive-test) before deciding on a refund.**

Confidence key used below: **[code]** read in source, **[vela]** Arm's
compiler estimate, **[calc]** arithmetic from the above, **[unverified]**
needs a datasheet or hardware.

## Hardware path

```
PAG7982J1 ──8-bit parallel, PCLK──▶ B1 LP-CPI ──DMA──▶ DTCM/SRAM ──▶ M55 (prep) ──▶ Ethos-U55-128
 640×480 BGGR8, global shutter       (HE domain)          2 MB total       160 MHz       128 MAC/cycle @160 MHz
```

| Part | Fact | Source |
|------|------|--------|
| Sensor | PixArt **PAG7982J1**, 640×480 colour, global shutter, 81.2° HFOV, "40 mW at full frame rate" | hardware manual **[code]** |
| Sensor link | I²C control @0x40; **8-bit parallel** to `lpcam`; `pixclk = 24 MHz`; app overlay sets `frame-rate = 30` | `boards/arm/halo/halo.dts`, `applications/halo/boards/halo.overlay` **[code]** |
| Sensor modes in driver | only 640×480 BGGR8; QVGA/QQVGA entries commented out; a PixArt-authored **"48M 640x480"** register table is commented out | `drivers/video/pag7982.c` **[code]** |
| Frame period | `R_FRAMETIME = pixclk / (2·fps)`, i.e. settable per build | `pag7982.c` **[code]** |
| Sensor features | window-of-interest crop (`R_ISP_WOI_*`), analog/ROI skip control (`R_ANALOG_SKIP_CTL`, `CMD_AND_ROISKIP_CTL`), on-chip AE | `pag7982.h` **[code]** |
| SoC | Alif Balletto B1, single Cortex-M55 "HE" core @ **160 MHz**, Ethos-U55 **128 MAC/cycle** (→ 41 GOP/s at 160 MHz; Alif's "46 GOPS" assumes a higher clock) | `halo_defconfig`, B1 dtsi **[code]** |
| Camera IF | LP-CPI, max 8 data bits, snapshot capture re-armed per frame by ISR → work queue | `drivers/video/video_alif.c` **[code]** |
| SRAM | 2 MB, all TCM: ITCM 512 KB + DTCM 856 KB + 640 KB region holding the video buffer pool. **No external RAM.** | `halo.dts` **[code]** |
| MRAM | 1.8 MB: MCUboot 128 KB, **two 780 KB image slots** (A/B OTA), 128 KB storage, 224 KB SE | `halo.dts` **[code]** |
| Stock firmware size | 0.8.17 OTA image = 586,732 B (573 KB) → ~207 KB free in a slot | release asset **[code]** |
| Battery | 300 mAh × 3.7 V ≈ 1.1 Wh | hardware manual **[code]** |

## Why the community sees single-digit fps

`modules/halo/src/lua_camera.c` builds a *still camera*:

1. One buffer (`CONFIG_VIDEO_BUFFER_POOL_NUM_MAX=1`).
2. Every `frame.camera.capture()` **starts the stream, throws away 3 warm-up
   frames, then stops the stream**. At 30 fps that's ≥100 ms before any
   processing.
3. Software debayer + black level + white balance + **JPEG encode on the M55**
   (libmpix).
4. A Lua loop reads the 16–80 KB JPEG and sends it over BLE in MTU-sized chunks.
5. The capture thread polls with 10 ms sleeps.

None of that bounds what the sensor + CPI + NPU can do when frames never leave
the device. The NPU is unused by the stock app; the Ethos-U driver, TFLM and
an inference shell are in the tree (`subsys/modules/ethosu`,
`lib/ethosu_utils`).

The hardware path *can* stream continuously: `samples/halo/pag7982` enqueues
buffers and dequeues timestamped frames back-to-back at the overlay's 30 fps.

## Sensor → SoC bus ceiling **[calc]**

One BGGR8 byte per pixel, one byte per PCLK on the 8-bit bus. These are upper
bounds with zero blanking; real line/frame blanking costs some extra.

| Readout | Bytes/frame | 24 MHz (shipped) | 48 MHz (PixArt table) |
|---------|------------:|-----------------:|----------------------:|
| 640×480 | 307,200 | 12.8 ms → **≤ 78 fps** | 6.4 ms → ≤ 156 fps |
| 640×240 crop (horizon band) | 153,600 | 6.4 ms → ≤ 156 fps | ≤ 312 fps |
| 320×240 skip | 76,800 | 3.2 ms → ≤ 312 fps | ≤ 625 fps |

Consequences:

- Setting `frame-rate = <120>` at 24 MHz asks for an 8.33 ms frame period
  while VGA readout alone takes 12.8 ms. **It cannot work.** Expect either a
  clamped rate or corrupt frames.
- "120 fps VGA" fits a 48 MHz PCLK with ~23% blanking. That matches PixArt's
  own commented-out table, but it's an inference: **[unverified]** until the
  PAG7982J1 datasheet or a test confirms it. Check the footnote on PixArt's
  120 fps claim; such figures are often quoted for a sub-window or skip mode.
- **[unverified]** maximum PCLK the B1 **LP-CPI** accepts. The DMA side is not
  the issue: 48 MB/s is ~5% of DTCM bandwidth. The open question is input
  sampling of the parallel bus.

PRIMAL doesn't need VGA. The trunk eats 96×160 or 120×160. A cropped or
skipped readout makes 120 fps a **24 MHz** problem, not a 48 MHz one, and cuts
buffer memory 2–4×.

## NPU estimates **[vela]**

Ethos-U55-128 @ 160 MHz, arena in DTCM, weights in MRAM, using Alif's own
measured HE DTCM and MRAM timings (`tools/halo_npu/b1_vela.ini`).

**Sanity check:** the same flow gives **8.96 ms** for MobileNetV2-1.0/224 on
Alif's HP NPU config (U55-256 @ 400 MHz), against Alif's published ~8 ms.

| Model (int8, random weights) | MMAC | Arena KiB | Weights KiB | ms | fps |
|------------------------------|-----:|----------:|------------:|---:|----:|
| PRIMAL trunk 96×160 RGB (as in `train/model.py`, ReLU) | 43 | 254 | 571 | **2.96** | **338** |
| same, GELU | 43 | 254 | 572 | 3.16 | 316 |
| trunk 120×160 RGB (4:3, 4× subsample) | 55 | 299 | 651 | 3.70 | 270 |
| trunk 120×160 RGB, width 48 | 120 | 555 | 1258 | 8.34 | 120 |
| trunk 240×320 RGB | 218 | 974 | 1611 | 14.4 | 70 |
| raw-Bayer QVGA 240×320 → learned demosaic → trunk | 63 | 472 | 653 | 4.18 | 239 |
| raw-Bayer VGA 480×640 → … → trunk | 65 | **1503** | 653 | 6.58 | 152 |
| alignment head, full map N=1800, K=8 | 271 | 366 | 99 | 20.7 | 48 |
| alignment head, window N=256, K=8 | 39 | 77 | 99 | 3.00 | 333 |
| MobileNetV2-0.35 96×160 (reference) | 18 | 242 | 665 | 4.63 | 216 |
| MobileNetV2-1.0 224×224 (reference) | 300 | 1474 | 2501 | 35.5 | 28 |

All models map 100% onto the NPU (0 CPU fallback ops). Vela is an estimate;
budget **+20–30%** for TFLM invoke overhead and estimator error until it is
measured on the device.

What this means:

- **The trunk is not the bottleneck.** About 3–4 ms per frame leaves room at
  120 fps.
- **The full-map alignment head is.** At 20.7 ms it can't run per frame at
  120 Hz. Run it on a window around the filter's prior (N=256 → 3 ms), or run
  the full head at 15–30 Hz over embeddings produced at 120 Hz. Both fit.
- **GroupNorm must go** before deployment: it has no Ethos-U lowering. Fold
  BatchNorm or drop the norm. GELU is fine.
- **Raw-Bayer VGA input doesn't fit** (1.5 MB arena). Either bin on the M55
  first, or have the sensor skip/crop and feed QVGA mosaic straight to the NPU.

## Memory budget **[calc]**

**SRAM, 2 MB total.** The stock app uses most of it (Lua heaps, AEC, JPEG
buffer 115 KB, libmpix 80 KB, display), so a PRIMAL build replaces the
camera/JPEG path and probably trims Lua and audio.

| Item | VGA readout | Cropped/skipped QVGA readout |
|------|------------:|-----------------------------:|
| CPI buffers (×3 for capture ‖ prep ‖ spare) | 900 KB | 225 KB |
| NPU arena (trunk 120×160) | 299 KB | 299–472 KB |
| Reference map, 1800 × 128 int8 | 230 KB | 230 KB |
| Head arena (windowed) | 77 KB | 77 KB |
| **Total** | **~1.5 MB** — fits only after stripping the stock app | **~0.8–1.0 MB** — comfortable |

**MRAM.** Trunk weights are 571–651 KB, and the stock image leaves ~207 KB free
in its 780 KB slot.

- Shrink the weights. Swapping `Flatten → Linear` (246 K params, 43% of the
  trunk) for global pooling + linear brings the trunk to ~350 KB.
- And/or strip the image.
- Or load weights into SRAM over BLE at session start: ~600 KB at ~100 KB/s
  is ~6 s.
- Collapsing the A/B slots works but gives up OTA rollback. Avoid it.

## Per-frame budget at 120 fps (8.33 ms) **[calc]**

| Stage | Cost | Runs on | Overlaps? |
|-------|------|---------|-----------|
| Sensor readout + CPI DMA | 3.2–6.4 ms | sensor + DMA | yes, hardware |
| Prep: 2×2 bin / uint8→int8 (Helium) | ~0.1–2 ms (est.) | M55 | yes, with NPU of previous frame |
| Trunk | ~3–4 ms | NPU | — |
| Windowed head | ~3 ms | NPU | can run at lower rate |
| Estimator / filter, display | <1 ms | M55 | yes |

NPU duty at 120 fps is about 40–80%, depending on how often the head runs.
**60 fps is comfortable; 120 fps is feasible but tight.**

**Power** isn't a blocker: even at a pessimistic 0.5 W total, 1.1 Wh lasts
over 2 h, against a 25 min session. Not measured.

## Firmware changes needed

1. A continuous capture mode beside `capture()`:
   - ≥2–3 buffers (`CONFIG_VIDEO_BUFFER_POOL_NUM_MAX`)
   - stream started once
   - dequeue with `K_FOREVER` or `k_poll`, not 10 ms sleeps
   - no JPEG
2. Sensor configuration:
   - `frame-rate` 60/90/120
   - a cropped or skipped window via the WOI/skip registers (needs the
     PAG7982 register datasheet, or PixArt/Brilliant support)
   - or the 48 MHz table
3. Set `wait-vsync` on `&lpcam`. A late re-arm then drops a whole frame
   instead of tearing one.
4. Timestamp in the CPI ISR with the cycle counter. The driver currently
   stamps `k_uptime_get_32()` (1 ms resolution) in a work item.
5. TFLM + Ethos-U inference thread. The pieces are already in the tree.

## Decisive test

About a day of work, and reversible: OTA images test-boot and auto-revert
(`applications/halo/FLASHING.md`).

1. Add a Lua call, e.g. `frame.camera.bench(n)`, that streams `n` frames
   continuously with 3 buffers. Have it return:
   - min/mean/max frame interval
   - count of `INTR_INFIFO_OVERRUN` / `OUTFIFO_OVERRUN` / short frames
2. Run it at `frame-rate` 30 → 60 → 75 at 24 MHz VGA. This finds the
   practical 24 MHz ceiling, expected 60–70 fps.
3. Swap in the commented "48M" table and try 90 → 120.
   - Overruns or garbage → the LP-CPI or the module can't take 48 MHz.
   - Clean frames → 120 fps VGA confirmed.
4. Independently, ask PixArt or Brilliant for the PAG7982J1 register
   datasheet (window/skip registers, max PCLK, frame rate per mode). That
   unlocks the cropped-readout route.
5. Run `trunk_120x160` through TFLM on the NPU and time `Invoke()` to replace
   the Vela estimate.

**Reading the results:**

| Outcome | What it means |
|---------|---------------|
| Step 2 reaches ~60 fps | 60 Hz on-glasses embedding is established with no unverified parts |
| Step 3 or a cropped mode reaches 120 | 120 Hz is established |
| Step 2 stalls far below 60 with clean timing | Unexpected; the specific limit is worth chasing before any refund decision |

## Reproduce the NPU numbers

```bash
pip install tensorflow-cpu ethos-u-vela
python tools/halo_npu/models.py build
python tools/halo_npu/estimate.py build
```

## Sources

- Firmware: <https://github.com/brilliantlabsAR/halo-firmware>
  - `drivers/video/pag7982.{c,h}`
  - `modules/halo/src/lua_camera.c`
  - `boards/arm/halo/halo.dts`
  - `applications/halo/boards/halo.{overlay,conf}`
  - `samples/halo/pag7982`
- Zephyr fork, `drivers/video/video_alif.{c,h}`, `dts/arm/alif/balletto_rtss_common.dtsi`: <https://github.com/brilliantlabsAR/halo-zephyr-alif>
- Halo hardware manual: <https://github.com/brilliantlabsAR/docs/blob/main/halo/hardware.md>
- Alif Vela timings, `scripts/vela/ensemble_vela.ini`: <https://github.com/alifsemi/alif_ml-embedded-evaluation-kit>
- Not reachable from the research environment (egress policy):
  - pixart.com (PAG7982J1 page)
  - alifsemi.com (B1 datasheet)
  - docs.brilliant.xyz
