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
local smoothLat, smoothSlope
local waves, wavesSeed
local lastWander = 2.5

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
  -- systemTime arrives as a 64-bit cdata integer, which math functions reject.
  if seed == 0 then seed = tonumber(ac.getSim().systemTime) % 100000 end
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

local KNEE_M = 0.6    -- width of the soft zone before the edge limit
local SMOOTH_M = 6.0  -- distance over which the applied offset and heading are smoothed

--- How far the camera can go in one direction before leaving edge_limit, searched
--- up to `reach`. Track coordinates, so independent of which way `side` points.
local function edgeRoom(car, direction, reach)
  if math.abs(ac.worldCoordinateToTrack(place(car, direction * reach)).x) <= cfg.edge_limit then return reach end
  local lo, hi = 0, reach
  for _ = 1, 12 do
    local mid = (lo + hi) / 2
    if math.abs(ac.worldCoordinateToTrack(place(car, direction * mid)).x) > cfg.edge_limit then hi = mid else lo = mid end
  end
  return lo
end

--- A soft limit instead of a hard clamp: offsets pass unchanged until KNEE_M from
--- the edge, then compress smoothly toward it with matching slope, so reaching the
--- edge bends the path instead of kinking it.
local function softLimit(car, requested)
  if requested == 0 then return 0, false end
  local direction = requested > 0 and 1 or -1
  local want = math.abs(requested)
  local room = edgeRoom(car, direction, want + KNEE_M)
  local knee = room - KNEE_M
  if want <= knee then return requested, false end
  local limited = room - KNEE_M * math.exp(-(want - knee) / KNEE_M)
  return direction * math.max(0, math.min(want, limited)), true
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
  smoothLat = nil
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
    if grabbed then camera, smoothLat = grabbed, nil else lastError = 'grab: ' .. tostring(err) end
  elseif not want and camera then
    camera:dispose()
    camera = nil
  end
  if not camera then return end

  local car = ac.getCar(0)
  -- Distance driven, not track position, drives the wander, so each lap takes a
  -- different line through the same corner instead of repeating one.
  local ds = math.abs(car.speedKmh) / 3.6 * dt
  distance = distance + ds
  local offset = wander(distance)
  local target, wasClamped = softLimit(car, cfg.lateral_m + offset)

  -- Smooth the applied offset over distance, and take the heading from the path
  -- the camera actually follows, so the edge limit can never snap either of them.
  if smoothLat == nil then smoothLat, smoothSlope = target, 0 end
  local previous = smoothLat
  local alpha = 1 - math.exp(-ds / SMOOTH_M)
  smoothLat = smoothLat + alpha * (target - smoothLat)
  if ds > 1e-4 then smoothSlope = smoothSlope + alpha * ((smoothLat - previous) / ds - smoothSlope) end
  local lateral = smoothLat
  local position = place(car, lateral)
  local x = ac.worldCoordinateToTrack(position).x

  local look = car.look
  local yaw = 0
  if cfg.wander_yaw == 1 then
    look = car.look - car.side * smoothSlope
    yaw = math.deg(math.atan(smoothSlope))
  end
  camera.transform.position:set(position)
  camera.transform.look:set(look)
  camera.transform.up:set(car.up)
  camera.fov = cfg.fov_deg
  -- A grabbed camera replaces every camera parameter, not just the pose. Pure
  -- does much of its look through camera exposure, so pass exposure and depth of
  -- field through from AC, or the image renders flat, like stock AC.
  camera.exposure = camera.exposureOriginal
  camera.dofFactor = camera.dofFactorOriginal
  camera.dofDistance = camera.dofDistanceOriginal
  camera.ownShare = 1
  applied, trackX, clamped, wanderNow, yawNow = lateral, x, wasClamped, offset, yaw

  local key = renderKey()
  local frame = tonumber(sim.replayCurrentFrame) or -1
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

local CFG_KEYS = { 'enabled', 'replay_only', 'lateral_m', 'wander_m', 'wander_len_m', 'wander_yaw', 'wander_seed',
  'forward_m', 'height_m', 'fov_deg', 'edge_limit', 'weather', 'rain' }

--- The toggles write rig.txt, so the file stays the one source of settings and the
--- next re-read does not undo a click.
local function saveConfig()
  local lines = {}
  for _, key in ipairs(CFG_KEYS) do lines[#lines + 1] = key .. ' = ' .. tostring(cfg[key]) end
  io.save(cfgPath, table.concat(lines, '\n') .. '\n')
end

function script.windowMain(dt)
  if cfg.wander_m > 0 then lastWander = cfg.wander_m end
  if ui.checkbox('Rig on', cfg.enabled == 1) then
    cfg.enabled = 1 - cfg.enabled
    saveConfig()
  end
  if ui.checkbox('Live driving too (off: replays only)', cfg.replay_only == 0) then
    cfg.replay_only = 1 - cfg.replay_only
    saveConfig()
  end
  if ui.checkbox(string.format('Wander (%.1f m)', lastWander), cfg.wander_m > 0) then
    cfg.wander_m = cfg.wander_m > 0 and 0 or lastWander
    saveConfig()
  end
  ui.text('Close this window before recording: it would be in the footage.')
  ui.separator()
  ui.text(string.format('enabled %d   replay only %d   holding camera %s', cfg.enabled, cfg.replay_only,
    tostring(camera ~= nil)))
  ui.text(string.format('lateral %+.2f + wander %+.2f   applied %+.2f m   yaw %+.1f   track x %+.3f %s',
    cfg.lateral_m, wanderNow, applied, yawNow, trackX, clamped and 'CLAMPED' or ''))
  ui.text(string.format('forward %.2f m   height %.2f m   fov %.1f   weather %d   rain %.2f', cfg.forward_m,
    cfg.height_m, cfg.fov_deg, cfg.weather, cfg.rain))
  ui.text(string.format('clamped %d of %d frames   log rows %d   saved %s', clampFrames, heldFrames, #rows,
    lastSaveOk and 'ok' or 'FAILED'))
  if lastError then ui.textWrapped('ERROR ' .. lastError) end
end
