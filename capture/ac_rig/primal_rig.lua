--- Camera rig for rendering from controlled viewpoints and conditions.
---
--- The bot drives one line, so every recorded lap shares a viewpoint. Mounting the
--- camera at a sideways offset from the car gives other lines: a fixed offset for
--- evaluation, where separation must be known exactly, or a smooth random wander
--- for training, so one drive covers many lines. It works live, while the bot
--- drives, or on a replay. Replays can also be re-rendered under different
--- weather, since CSP lets a script override replay conditions.
---
--- The camera stays rigid to the car body, so its pitch and roll come through as
--- they would on a real mount. Labels need no help from here: the timecode app
--- encodes the rendering camera's own track position.
---
--- Settings come from rig.txt (key=value, re-read twice a second), so they change
--- without restarting AC or putting anything on screen. Each render gets its own
--- log file, and no file is ever overwritten.

local folder = ac.getFolder(ac.FolderID.ScriptOrigin)
local cfgPath = folder .. '/rig.txt'
local cfg = {
  enabled = 0,
  replay_only = 1,
  lateral_m = 0.0,       -- fixed offset; positive is right, matching track x where +1 is the right edge
  wander_m = 0.0,        -- peak of the smooth random wander added to lateral_m; 0 switches it off
  wander_len_m = 200.0,  -- distance over which the wander swings; long enough to look like a line change
  wander_yaw = 1,        -- turn the camera along the wander's direction, as a car changing line would
  wander_seed = 0,       -- 0 picks a new seed per launch
  forward_m = 2.2,       -- ahead of the car origin; past the front so the body stays out of frame
  height_m = 1.15,       -- above the car origin, which sits at road level
  fov_deg = 60.0,        -- vertical, matching AC's own cameras
  edge_limit = 0.85,     -- max |track x|, where +-1 are the track edges
  weather = -1,          -- ac.WeatherType to force during replays; -1 keeps the recorded weather
  rain = 0.0,            -- 0..1, applied to rain intensity, wetness and puddles when weather is forced
}

local HEADER = 'clock_s,replay_frame,car_spline,cam_trk_s,car_trk_x,cam_trk_x,cam_trk_h,side_left,side_right,'
  .. 'car_lat_m,cam_lat_m,requested_lateral_m,applied_lateral_m,clamped,weather,rain,'
  .. 'distance_m,wander_m,yaw_deg\n'
local LOG_EVERY_S = 1 / 30

local camera
local conditions
local overriding = false
local lastError
local sinceConfig, sinceLog, sinceSave, clock = 1, 0, 0, 0
local rows, logPath, logKey = {}, nil, nil
local lastSaveOk = true
local lastFrame = -1
local applied, trackX, clamped, wanderNow, yawNow = 0, 0, false, 0, 0
local clampFrames, heldFrames = 0, 0
local distance = 0
local waves, wavesSeed

local function loadConfig()
  local text = io.load(cfgPath)
  if not text then return end
  for key, value in text:gmatch('([%w_]+)%s*=%s*([%-%+%.%w]+)') do
    local number = tonumber(value)
    if number ~= nil and cfg[key] ~= nil then cfg[key] = number end
  end
end

--- A fixed hash of seed and index into [0, 1). Used instead of math.random,
--- which the CSP app sandbox may not provide.
local function hash01(seed, i)
  local v = math.sin(seed * 12.9898 + i * 78.233) * 43758.5453
  return v - math.floor(v)
end

--- Three sines at incommensurate wavelengths around wander_len_m, with seeded
--- phases, so the wander never repeats and is smooth at every scale a clip sees.
local function makeWaves()
  local seed = cfg.wander_seed
  if seed == 0 then seed = ac.getSim().systemTime % 100000 end
  waves = {}
  local shape = { { 0.62, 1.0 }, { 1.0, 0.7 }, { 1.73, 0.5 } }
  for i = 1, #shape do
    waves[i] = {
      k = 2 * math.pi / (cfg.wander_len_m * shape[i][1]),
      weight = shape[i][2],
      phase = hash01(seed, i) * 2 * math.pi,
    }
  end
  wavesSeed = cfg.wander_seed
end

--- Wander offset and its slope (metres sideways per metre driven) at a distance.
local function wander(d)
  if cfg.wander_m == 0 then return 0, 0 end
  if not waves or wavesSeed ~= cfg.wander_seed then makeWaves() end
  local sum, slope, total = 0, 0, 0
  for _, w in ipairs(waves) do
    sum = sum + w.weight * math.sin(w.k * d + w.phase)
    slope = slope + w.weight * w.k * math.cos(w.k * d + w.phase)
    total = total + w.weight
  end
  return cfg.wander_m * sum / total, cfg.wander_m * slope / total
end

local function place(car, lateral)
  -- AC's side vector points left, so subtract it to make positive offsets go right.
  return car.position + car.look * cfg.forward_m - car.side * lateral + car.up * cfg.height_m
end

--- Shrinks the sideways offset where the requested one would leave the tarmac.
--- Works in track coordinates, so it is independent of which way `side` points.
local function clampedPlacement(car, requested)
  local position = place(car, requested)
  local x = ac.worldCoordinateToTrack(position).x
  if math.abs(x) <= cfg.edge_limit or requested == 0 then
    return position, requested, x, false
  end
  local lo, hi = 0, requested
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
  return string.format('lat%+.2f_wan%.1f_w%d_r%.2f', cfg.lateral_m, cfg.wander_m, cfg.weather, cfg.rain)
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
  -- Distance driven, not track position, drives the wander, so each lap takes a
  -- different line through the same corner instead of repeating one.
  distance = distance + math.abs(car.speedKmh) / 3.6 * dt
  local offset, slope = wander(distance)
  local position, lateral, x, wasClamped = clampedPlacement(car, cfg.lateral_m + offset)

  local look = car.look
  local yaw = 0
  if cfg.wander_yaw == 1 and not wasClamped and slope ~= 0 then
    look = car.look - car.side * slope
    yaw = math.deg(math.atan(slope))
  end
  camera.transform.position:set(position)
  camera.transform.look:set(look)
  camera.transform.up:set(car.up)
  camera.fov = cfg.fov_deg
  camera.ownShare = 1
  applied, trackX, clamped, wanderNow, yawNow = lateral, x, wasClamped, offset, yaw

  local key = renderKey()
  local frame = sim.replayCurrentFrame or -1
  if key ~= logKey or frame < lastFrame - 60 then startLog(key) end
  lastFrame = frame
  heldFrames = heldFrames + 1
  if wasClamped then clampFrames = clampFrames + 1 end
  sinceLog = sinceLog + dt
  sinceSave = sinceSave + dt
  if sinceLog >= LOG_EVERY_S and #rows < 60000 then
    sinceLog = 0
    local camTrk = ac.worldCoordinateToTrack(position)
    local carTrk = ac.worldCoordinateToTrack(car.position)
    local camLat, sides = metresFromMiddle(camTrk)
    local carLat = metresFromMiddle(carTrk)
    rows[#rows + 1] = string.format(
      '%.3f,%s,%.6f,%.6f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d,%.2f,%.1f,%.3f,%.2f',
      clock, tostring(frame), car.splinePosition, camTrk.z, carTrk.x, camTrk.x, camTrk.y, sides.x, sides.y,
      carLat, camLat, cfg.lateral_m + offset, lateral, wasClamped and '1' or '0', cfg.weather, cfg.rain,
      distance, offset, yaw)
  end
  if sinceSave >= 2.0 then
    sinceSave = 0
    lastSaveOk = io.save(logPath, HEADER .. table.concat(rows, '\n') .. '\n')
  end
end

local failedKey

--- A failure releases the camera rather than leaving it frozen, so a broken rig can
--- never record a stationary view, and it stays off until rig.txt changes. The
--- full error goes to rig_error.txt, since the window truncates it.
function script.update(dt)
  if failedKey then
    sinceConfig = sinceConfig + dt
    if sinceConfig >= 0.5 then
      sinceConfig = 0
      loadConfig()
    end
    if renderKey() .. cfg.enabled == failedKey then return end
    failedKey = nil
  end
  local ok, err = pcall(step, dt)
  if not ok then
    lastError = 'update: ' .. tostring(err)
    io.save(folder .. '/rig_error.txt', lastError .. '\n')
    if camera then
      pcall(function() camera:dispose() end)
      camera = nil
    end
    failedKey = renderKey() .. cfg.enabled
  end
end

function script.windowMain(dt)
  ui.text(string.format('enabled %d   replay only %d   holding camera %s', cfg.enabled, cfg.replay_only,
    tostring(camera ~= nil)))
  ui.text(string.format('lateral %+.2f + wander %+.2f   applied %+.2f m   yaw %+.1f   track x %+.3f %s',
    cfg.lateral_m, wanderNow, applied, yawNow, trackX, clamped and 'CLAMPED' or ''))
  ui.text(string.format('forward %.2f m   height %.2f m   fov %.1f   weather %d   rain %.2f', cfg.forward_m,
    cfg.height_m, cfg.fov_deg, cfg.weather, cfg.rain))
  ui.text(string.format('clamped %d of %d frames   log rows %d   saved %s', clampFrames, heldFrames, #rows,
    lastSaveOk and 'ok' or 'FAILED'))
  if lastError then ui.text('ERROR ' .. lastError) end
end
