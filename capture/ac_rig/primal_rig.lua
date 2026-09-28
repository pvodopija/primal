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
  look = 0,              -- turn the camera into corners, as a driver's head does
  look_ahead_min_m = 15, -- how far along the track the head aims, drawn per render from this range
  look_ahead_max_m = 20,
  look_gain_min = 0.3,   -- fraction of the angle to that point the head turns, drawn per render
  look_gain_max = 0.8,
  look_max_deg = 35,     -- the head never turns further than this into a corner
  glance_deg = 8,        -- peak of the occasional short glance aside while looking; 0 switches glances off
  glance_every_s = 10,   -- mean time between glances
  replay_now = 0,        -- set to 1 to open the replay, whose bar has Save Replay; resets itself to 0
}

-- yaw_deg is the camera's whole turn relative to the car, positive right; head_yaw_deg
-- is the part from looking into corners, including glance_deg, so the wander's
-- part is yaw_deg - head_yaw_deg.
local HEADER = 'clock_s,replay_frame,car_spline,cam_trk_s,car_trk_x,cam_trk_x,cam_trk_h,side_left,side_right,'
  .. 'car_lat_m,cam_lat_m,requested_lateral_m,applied_lateral_m,clamped,weather,rain,'
  .. 'distance_m,wander_m,yaw_deg,head_yaw_deg,glance_deg,look_gain,look_ahead_m\n'
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
local renders, launchSeed = 0, nil
local lookGain, lookAhead = 0, 0
local headSmooth, headNow, glanceNow = nil, 0, 0
local glanceStart, glanceLen, glanceAmp, glanceNext, glances = 0, 0, 0, nil, 0

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

local HEAD_TAU_S = 0.25  -- how quickly the head follows the corner, as a time constant

--- wander_seed when set, otherwise one seed per launch.
local function seed()
  if cfg.wander_seed ~= 0 then return cfg.wander_seed end
  launchSeed = launchSeed or tonumber(ac.getSim().systemTime) % 100000
  return launchSeed
end

--- Heading, relative to the car and positive right, of the point `lookAhead` metres
--- further along the track on the camera's own line. On a straight it is zero
--- wherever the camera sits; in a corner it points to where the road goes.
local function cornerDeg(car, position)
  local length = ac.getSim().trackLengthM
  if not (length and length > 0) then return 0 end
  local trk = ac.worldCoordinateToTrack(position)
  local s = trk.z + lookAhead / length
  local dir = ac.trackCoordinateToWorld(vec3(trk.x, trk.y, s - math.floor(s))) - position
  local deg = math.deg(math.atan2(-dir:dot(car.side), dir:dot(car.look)))
  if deg ~= deg then return 0 end
  return deg
end

--- An occasional short glance aside, out and back, at a random time, size and side.
local function glance(t)
  if cfg.glance_deg <= 0 or cfg.glance_every_s <= 0 then return 0 end
  -- Draws for glance n use hash indices 1000 + 4n .. 1000 + 4n + 3, apart from the wander's and the renders'.
  local function draw(k) return hash01(seed(), 1000 + 4 * glances + k) end
  if not glanceNext then glanceNext = t + cfg.glance_every_s * (0.5 + draw(3)) end
  if t >= glanceNext then
    glances = glances + 1
    glanceStart = t
    glanceLen = 0.6 + 0.8 * draw(0)
    glanceAmp = cfg.glance_deg * (0.4 + 0.6 * draw(1))
    if draw(2) < 0.5 then glanceAmp = -glanceAmp end
    glanceNext = t + glanceLen + cfg.glance_every_s * (0.5 + draw(3))
  end
  local u = (t - glanceStart) / glanceLen
  if glanceLen <= 0 or u >= 1 then return 0 end
  return glanceAmp * math.sin(math.pi * u) ^ 2
end

--- The head's turn this frame: a share of the corner angle, capped and smoothed
--- in time, plus any glance on top.
local function headYaw(car, position, dt)
  local target = lookGain * cornerDeg(car, position)
  target = math.max(-cfg.look_max_deg, math.min(cfg.look_max_deg, target))
  if headSmooth == nil then headSmooth = target end
  headSmooth = headSmooth + (1 - math.exp(-dt / HEAD_TAU_S)) * (target - headSmooth)
  glanceNow = glance(clock)
  return headSmooth + glanceNow
end

local function renderKey()
  return string.format('lat%+.2f_wan%.1f_w%d_r%.2f%s', cfg.lateral_m, cfg.wander_m, cfg.weather, cfg.rain,
    cfg.look == 1 and '_look' or '')
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
  -- A new head for every render, so re-rendering one replay gives different ones.
  -- Render n draws hash indices 10 + 2n and 11 + 2n, below the glances'.
  renders = renders + 1
  lookGain = cfg.look_gain_min + (cfg.look_gain_max - cfg.look_gain_min) * hash01(seed(), 10 + 2 * renders)
  lookAhead = cfg.look_ahead_min_m + (cfg.look_ahead_max_m - cfg.look_ahead_min_m) * hash01(seed(), 11 + 2 * renders)
  headSmooth = nil
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
  local key = renderKey()
  local frame = tonumber(sim.replayCurrentFrame) or -1
  if key ~= logKey or frame < lastFrame - 60 then startLog(key) end
  lastFrame = frame

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

  -- Turn about the car's up axis: along the wander's heading, then the head on top.
  local head = cfg.look == 1 and headYaw(car, position, dt) or 0
  if cfg.look ~= 1 then glanceNow = 0 end
  local yaw = head
  if cfg.wander_yaw == 1 then yaw = yaw + math.deg(math.atan(smoothSlope)) end
  local turn = math.rad(yaw)
  local look = car.look * math.cos(turn) - car.side * math.sin(turn)
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
  applied, trackX, clamped, wanderNow, yawNow, headNow = lateral, x, wasClamped, offset, yaw, head

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
      '%.3f,%s,%.6f,%.6f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%d,%.2f,%.1f,%.3f,%.2f,%.2f,%.2f,%.3f,%.1f',
      clock, tostring(frame), car.splinePosition, camTrk.z, carTrk.x, camTrk.x, camTrk.y, sides.x, sides.y,
      carLat, camLat, cfg.lateral_m + offset, lateral, wasClamped and '1' or '0', cfg.weather, cfg.rain,
      distance, offset, yaw, head, glanceNow, cfg.look == 1 and lookGain or 0, cfg.look == 1 and lookAhead or 0)
  end
  if sinceSave >= 2.0 then
    sinceSave = 0
    lastSaveOk = io.save(logPath, HEADER .. table.concat(rows, '\n') .. '\n')
  end
end

local failedKey
local saveConfig

--- Opens the replay on request from rig.txt, for unattended sessions: the pause
--- menu needs Escape, which a remote-control tool may keep for itself.
local function replayRequest()
  if cfg.replay_now ~= 1 then return end
  cfg.replay_now = 0
  saveConfig()
  if not ac.getSim().isReplayActive then
    local ok = ac.tryToToggleReplay(true, 0)
    io.save(folder .. '/replay_request.txt', (ok and 'opened' or 'not available') .. '\n')
  end
end

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
  pcall(replayRequest)
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
  'forward_m', 'height_m', 'fov_deg', 'edge_limit', 'weather', 'rain', 'look', 'look_ahead_min_m', 'look_ahead_max_m',
  'look_gain_min', 'look_gain_max', 'look_max_deg', 'glance_deg', 'glance_every_s', 'replay_now' }

--- The toggles write rig.txt, so the file stays the one source of settings and the
--- next re-read does not undo a click.
function saveConfig()
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
  if ui.checkbox('Look into corners', cfg.look == 1) then
    cfg.look = 1 - cfg.look
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
  if cfg.look == 1 then
    ui.text(string.format('head %+.1f deg (glance %+.1f)   gain %.2f   aiming %.1f m ahead', headNow, glanceNow,
      lookGain, lookAhead))
  end
  ui.text(string.format('clamped %d of %d frames   log rows %d   saved %s', clampFrames, heldFrames, #rows,
    lastSaveOk and 'ok' or 'FAILED'))
  if lastError then ui.textWrapped('ERROR ' .. lastError) end
end
