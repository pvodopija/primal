--- Camera rig for re-rendering replays from controlled viewpoints and conditions.
---
--- The bot drives one line, so every recorded lap shares a viewpoint. Mounting the
--- camera at a known sideways offset from the car turns one replay into several
--- virtual lines with exact, known separation, which is what the lines gate needs
--- and what real AC footage otherwise lacks. Replays can also be re-rendered under
--- different weather, since CSP lets a script override replay conditions.
---
--- The camera stays rigid to the car body, so its pitch and roll come through as
--- they would on a real mount. Labels need no help from here: the timecode app
--- encodes the rendering camera's own track position.
---
--- Settings come from rig.txt (key=value, re-read twice a second), so they change
--- between renders without restarting AC or putting anything on screen. Each
--- render gets its own log file, and no file is ever overwritten.

local folder = ac.getFolder(ac.FolderID.ScriptOrigin)
local cfgPath = folder .. '/rig.txt'
local cfg = {
  enabled = 0,
  replay_only = 1,
  lateral_m = 0.0,   -- positive is right, matching track x where +1 is the right edge
  forward_m = 2.2,   -- ahead of the car origin; past the front so the body stays out of frame
  height_m = 1.15,   -- above the car origin, which sits at road level
  fov_deg = 60.0,    -- vertical, matching AC's own cameras
  edge_limit = 0.85, -- max |track x|, where +-1 are the track edges
  weather = -1,      -- ac.WeatherType to force during replays; -1 keeps the recorded weather
  rain = 0.0,        -- 0..1, applied to rain intensity, wetness and puddles when weather is forced
}

local HEADER = 'clock_s,replay_frame,car_spline,cam_trk_s,car_trk_x,cam_trk_x,cam_trk_h,side_left,side_right,'
  .. 'car_lat_m,cam_lat_m,requested_lateral_m,applied_lateral_m,clamped,weather,rain\n'

local camera
local conditions
local overriding = false
local lastError
local sinceConfig, sinceLog, sinceSave, clock = 1, 0, 0, 0
local rows, logPath, logKey = {}, nil, nil
local lastSaveOk = true
local lastFrame = -1
local applied, trackX, clamped = 0, 0, false
local clampFrames, heldFrames = 0, 0

local function loadConfig()
  local text = io.load(cfgPath)
  if not text then return end
  for key, value in text:gmatch('([%w_]+)%s*=%s*([%-%+%.%w]+)') do
    local number = tonumber(value)
    if number ~= nil and cfg[key] ~= nil then cfg[key] = number end
  end
end

local function place(car, lateral)
  -- AC's side vector points left, so subtract it to make positive offsets go right.
  return car.position + car.look * cfg.forward_m - car.side * lateral + car.up * cfg.height_m
end

--- Shrinks the sideways offset where the requested one would leave the tarmac.
--- Works in track coordinates, so it is independent of which way `side` points.
local function clampedPlacement(car)
  local lateral = cfg.lateral_m
  local position = place(car, lateral)
  local x = ac.worldCoordinateToTrack(position).x
  if math.abs(x) <= cfg.edge_limit or lateral == 0 then
    return position, lateral, x, false
  end
  local lo, hi = 0, lateral
  for _ = 1, 10 do
    local mid = (lo + hi) / 2
    if math.abs(ac.worldCoordinateToTrack(place(car, mid)).x) > cfg.edge_limit then hi = mid else lo = mid end
  end
  position = place(car, lo)
  return position, lo, ac.worldCoordinateToTrack(position).x, true
end

--- Track x is -1 at the left edge and +1 at the right, so half the track width
--- converts it to metres from the middle. Checked against the 2.5 m rig test:
--- 0.46 of track x against a 5.4-5.6 m half-width.
local function metresFromMiddle(trk)
  local sides = ac.getTrackAISplineSides(trk.z - math.floor(trk.z))
  return trk.x * (sides.x + sides.y) / 2, sides
end

local function renderKey()
  return string.format('lat%+.2f_w%d_r%.2f', cfg.lateral_m, cfg.weather, cfg.rain)
end

--- One file per render: a render starts whenever settings change or the replay
--- is rewound, and a free index is picked so no earlier render is overwritten.
local function startLog(key)
  local n = 1
  repeat
    logPath = string.format('%s/rig_%s_%d.csv', folder, key, n)
    n = n + 1
  until not io.fileExists(logPath)
  logKey = key
  rows = {}
  clampFrames, heldFrames = 0, 0
end

local function applyConditions(active)
  if active and cfg.weather >= 0 then
    conditions = conditions or ac.getConditionsSet()
    ac.getConditionsSetTo(conditions)
    conditions.currentType = cfg.weather
    conditions.upcomingType = cfg.weather
    conditions.transition = 0
    conditions.rainIntensity = cfg.rain
    conditions.rainWetness = cfg.rain
    conditions.rainWater = cfg.rain
    ac.overrideReplayConditions(conditions)
    overriding = true
  elseif overriding then
    ac.overrideReplayConditions(nil)
    overriding = false
  end
end

local function step(dt)
  clock = clock + dt
  sinceConfig = sinceConfig + dt
  if sinceConfig >= 0.5 then
    sinceConfig = 0
    loadConfig()
  end

  local sim = ac.getSim()
  local want = cfg.enabled == 1 and (cfg.replay_only == 0 or sim.isReplayActive)
  applyConditions(want and sim.isReplayActive)
  if want and not camera then
    local grabbed, err = ac.grabCamera('primal rig')
    if grabbed then camera = grabbed else lastError = 'grab: ' .. tostring(err) end
  elseif not want and camera then
    camera:dispose()
    camera = nil
  end
  if not camera then return end

  local car = ac.getCar(0)
  local position, lateral, x, wasClamped = clampedPlacement(car)
  camera.transform.position:set(position)
  camera.transform.look:set(car.look)
  camera.transform.up:set(car.up)
  camera.fov = cfg.fov_deg
  camera.ownShare = 1
  applied, trackX, clamped = lateral, x, wasClamped

  local key = renderKey()
  local frame = sim.replayCurrentFrame
  if key ~= logKey or frame < lastFrame - 60 then startLog(key) end
  lastFrame = frame
  heldFrames = heldFrames + 1
  if wasClamped then clampFrames = clampFrames + 1 end
  sinceLog = sinceLog + dt
  sinceSave = sinceSave + dt
  if sinceLog >= 0.1 and #rows < 40000 then
    sinceLog = 0
    local camTrk = ac.worldCoordinateToTrack(position)
    local carTrk = ac.worldCoordinateToTrack(car.position)
    local camLat, sides = metresFromMiddle(camTrk)
    local carLat = metresFromMiddle(carTrk)
    rows[#rows + 1] = string.format('%.3f,%s,%.6f,%.6f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d,%.2f',
      clock, tostring(frame), car.splinePosition, camTrk.z, carTrk.x, camTrk.x, camTrk.y, sides.x, sides.y,
      carLat, camLat, cfg.lateral_m, lateral, wasClamped and '1' or '0', cfg.weather, cfg.rain)
  end
  if sinceSave >= 2.0 then
    sinceSave = 0
    lastSaveOk = io.save(logPath, HEADER .. table.concat(rows, '\n') .. '\n')
  end
end

function script.update(dt)
  local ok, err = pcall(step, dt)
  if not ok then lastError = 'update: ' .. tostring(err) end
end

function script.windowMain(dt)
  ui.text(string.format('enabled %d   replay only %d   holding camera %s', cfg.enabled, cfg.replay_only,
    tostring(camera ~= nil)))
  ui.text(string.format('lateral requested %+.2f m   applied %+.2f m   track x %+.3f %s', cfg.lateral_m, applied,
    trackX, clamped and 'CLAMPED' or ''))
  ui.text(string.format('forward %.2f m   height %.2f m   fov %.1f   weather %d   rain %.2f', cfg.forward_m,
    cfg.height_m, cfg.fov_deg, cfg.weather, cfg.rain))
  ui.text(string.format('clamped %d of %d frames   log rows %d   saved %s', clampFrames, heldFrames, #rows,
    lastSaveOk and 'ok' or 'FAILED'))
  if lastError then ui.text('ERROR ' .. lastError) end
end
