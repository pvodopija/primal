--- Locamotif timecode overlay.
---
--- Draws a 2x27 grid of pure black / pure white cells encoding a render-frame
--- counter and the track position of the camera that rendered the frame. Every
--- captured video frame therefore carries its own exact label, which removes
--- screen-capture latency, frame drops and clock skew as sources of label error.
---
--- Wire format (must stay identical to ml/capture/timecode.py):
---   cells are row-major, 27 per row, 2 rows
---   cell  0        black   low luminance reference
---   cell  1        white   high luminance reference
---   cell  2        black   finder
---   cell  3        white   finder
---   cells 4..51    48 data bits, LSB first
---   cell 52        white   tail finder
---   cell 53        black   tail finder
---
--- Data bits:
---   bits  0..21    frame counter, LSB first, wraps at 2^22
---   bits 22..41    camera track progress as round(clamp(s,0,1) * (2^20 - 1)), LSB first
---   bits 42..47    checksum; bit j = parity of payload bits j, j+6, ... j+36

local CELL = 14
local COLS = 27
local ROWS = 2
local N_CELLS = COLS * ROWS

local COUNTER_BITS = 22
local SPLINE_BITS = 20
local CHECK_BITS = 6
local PAYLOAD_BITS = COUNTER_BITS + SPLINE_BITS
local DATA_BITS = PAYLOAD_BITS + CHECK_BITS

local COUNTER_WRAP = 2 ^ COUNTER_BITS
local SPLINE_MAX = 2 ^ SPLINE_BITS - 1

-- Flip to true to print the numbers next to the grid while checking the app is
-- alive, then flip back before recording. The text sits inside the same window
-- rect that the decoder crops away, so leaving it on is harmless but pointless.
local SHOW_TEXT = false

local BLACK = rgbm(0, 0, 0, 1)
local WHITE = rgbm(1, 1, 1, 1)

local counter = 0
local bits = {}
for i = 1, DATA_BITS do bits[i] = 0 end

-- Per-frame telemetry keyed by the same counter the barcode carries, so it joins
-- onto exactly the frames a capture kept with no clock involved. Each row also
-- carries AC's own times: sim_ms for the frame and phys_ms for the physics state
-- it shows, so sensor latency can be simulated from ground truth. One folder per
-- launch, written in small chunks; nothing is ever overwritten.
local LOG_FRAMES = true
local LOG_FLUSH_FRAMES = 120
local LOG_HEADER = 'counter,sim_ms,phys_ms,cam_s,speed_kmh,acc_x_g,acc_y_g,acc_z_g,'
  .. 'vel_x,vel_y,vel_z,gyro_x,gyro_y,gyro_z,replay\n'
local logDir, logRows, logChunk, logError = nil, {}, 0, nil

local function flushLog()
  if #logRows == 0 then return end
  logChunk = logChunk + 1
  io.save(string.format('%s/%06d.csv', logDir, logChunk), LOG_HEADER .. table.concat(logRows, '\n') .. '\n')
  logRows = {}
end

local function logFrame(spline)
  local sim = ac.getSim()
  local car = ac.getCar(0)
  if not logDir then
    -- systemTime is a 64-bit cdata integer; string and math functions need a number.
    logDir = ac.getFolder(ac.FolderID.ScriptOrigin) .. '/frame_log/' .. tostring(tonumber(sim.systemTime))
    io.createDir(logDir)
  end
  local a, v, w = car.acceleration, car.localVelocity, car.localAngularVelocity
  logRows[#logRows + 1] = string.format('%d,%.3f,%.3f,%.7f,%.3f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%d',
    counter, tonumber(sim.time), tonumber(car.timestamp), spline, car.speedKmh,
    a.x, a.y, a.z, v.x, v.y, v.z, w.x, w.y, w.z, sim.isReplayActive and 1 or 0)
  if #logRows >= LOG_FLUSH_FRAMES then flushLog() end
end

--- Writes `count` bits of `value` into `bits` at 1-based `at`, LSB first.
local function writeBits(value, at, count)
  local v = value
  for i = 0, count - 1 do
    bits[at + i] = v % 2
    v = math.floor(v / 2)
  end
end

--- Checksum bit j is the parity of payload bits j, j+6, j+12, ... j+36.
--- Equivalent to XOR-folding the 42 payload bits into seven 6-bit chunks.
local function writeChecksum()
  for j = 0, CHECK_BITS - 1 do
    local sum = 0
    local k = j
    while k < PAYLOAD_BITS do
      sum = sum + bits[1 + k]
      k = k + CHECK_BITS
    end
    bits[1 + PAYLOAD_BITS + j] = sum % 2
  end
end

local function cellColor(index)
  if index == 0 or index == 2 or index == N_CELLS - 1 then return BLACK end
  if index == 1 or index == 3 or index == N_CELLS - 2 then return WHITE end
  return bits[index - 4 + 1] == 1 and WHITE or BLACK
end

function script.windowMain(dt)
  -- Incremented in the draw callback rather than script.update so the counter
  -- advances once per rendered frame, matching what the capture sees.
  counter = (counter + 1) % COUNTER_WRAP

  -- The label has to describe where the image was taken. The car's spline position
  -- is its centre, and the camera sits up to a couple of metres ahead of or behind
  -- that by an amount that differs per car and per camera, so encode the camera's
  -- own track position.
  local spline = ac.worldCoordinateToTrack(ac.getSim().cameraPosition).z
  -- AC reports slightly outside [0,1] around the timing line on some tracks.
  spline = spline - math.floor(spline)

  writeBits(counter, 1, COUNTER_BITS)
  writeBits(math.floor(spline * SPLINE_MAX + 0.5), 1 + COUNTER_BITS, SPLINE_BITS)
  writeChecksum()

  local origin = ui.getCursor()
  for index = 0, N_CELLS - 1 do
    local col = index % COLS
    local row = math.floor(index / COLS)
    local p1 = origin + vec2(col * CELL, row * CELL)
    ui.drawRectFilled(p1, p1 + vec2(CELL, CELL), cellColor(index))
  end

  ui.dummy(vec2(COLS * CELL, ROWS * CELL))

  -- After the barcode is drawn, and isolated: a logging failure switches logging
  -- off for the session rather than touching the label.
  if LOG_FRAMES and not logError then
    local ok, err = pcall(logFrame, spline)
    if not ok then logError = tostring(err) end
  end

  if SHOW_TEXT then
    ui.text(string.format('%d  s=%.6f', counter, spline))
  end
end
