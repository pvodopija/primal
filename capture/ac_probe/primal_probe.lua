--- Measurement probe for designing the capture camera rig.
---
--- Logs, ten times a second, where the active camera sits relative to the car and
--- relative to the track, alongside the car's own spline position and the track
--- width. Answers from numbers rather than memory:
---   * does the camera move between cars (cam_rel_*)
---   * how far is the image's viewpoint from the position the timecode encodes
---     (cam_trk_s against car_spline)
---   * where across the width the bot actually drives (car_trk_x)
---   * does spline position follow a replay (replay_* against car_spline)
---
--- Logs only while its window is visible: an open window would be burned into any
--- recording, so it can never be active during real capture.

local LOG_EVERY_S = 0.1
local SAVE_EVERY_S = 2.0
local MAX_ROWS = 30000

local path
local rows = {}
local clock, sinceLog, sinceSave = 0, 0, 0
local header
local lastError
local lastSaveOk = true

local function makeHeader()
  -- One file per car, so comparing two cars never overwrites the first.
  path = ac.getFolder(ac.FolderID.ScriptOrigin) .. '/probe_' .. tostring(ac.getCarID(0)) .. '.csv'
  local eyes = ac.getOnboardCameraDefaultParams(0)
  return string.format('# car=%s driver_eyes=%.4f,%.4f,%.4f onboard_pitch=%.2f\n',
      tostring(ac.getCarID(0)), eyes.position.x, eyes.position.y, eyes.position.z, eyes.pitch)
    .. 'clock_s,replay_active,replay_frame,replay_frames,playback_rate,camera_mode,drivable_mode,fov,'
    .. 'cam_rel_x,cam_rel_y,cam_rel_z,'
    .. 'car_spline,car_trk_x,car_trk_h,car_trk_s,'
    .. 'cam_trk_x,cam_trk_h,cam_trk_s,'
    .. 'side_left,side_right,'
    .. 'cam_fwd_x,cam_fwd_y,cam_fwd_z,car_look_x,car_look_y,car_look_z\n'
end

local function sample()
  local sim = ac.getSim()
  local car = ac.getCar(0)
  local rel = ac.getCameraPositionRelativeToCar()
  local carTrk = ac.worldCoordinateToTrack(car.position)
  local camTrk = ac.worldCoordinateToTrack(sim.cameraPosition)
  local sides = ac.getTrackAISplineSides(car.splinePosition)
  local fwd = ac.getCameraForward()
  rows[#rows + 1] = string.format(
    '%.3f,%s,%s,%s,%.3f,%s,%s,%.2f,%.4f,%.4f,%.4f,%.6f,%.4f,%.4f,%.6f,%.4f,%.4f,%.6f,%.3f,%.3f,'
      .. '%.4f,%.4f,%.4f,%.4f,%.4f,%.4f',
    clock, sim.isReplayActive and '1' or '0', tostring(sim.replayCurrentFrame),
    tostring(sim.replayFrames), sim.replayPlaybackRate, tostring(sim.cameraMode),
    tostring(sim.driveableCameraMode), sim.cameraFOV,
    rel.x, rel.y, rel.z,
    car.splinePosition, carTrk.x, carTrk.y, carTrk.z,
    camTrk.x, camTrk.y, camTrk.z,
    sides.x, sides.y,
    fwd.x, fwd.y, fwd.z, car.look.x, car.look.y, car.look.z)
end

function script.windowMain(dt)
  clock = clock + dt
  sinceLog = sinceLog + dt
  sinceSave = sinceSave + dt

  if not header then
    local ok, result = pcall(makeHeader)
    if ok then header = result else lastError = 'header: ' .. tostring(result) end
  end
  if header and sinceLog >= LOG_EVERY_S and #rows < MAX_ROWS then
    sinceLog = 0
    local ok, err = pcall(sample)
    if not ok then lastError = 'sample: ' .. tostring(err) end
  end
  if header and sinceSave >= SAVE_EVERY_S then
    sinceSave = 0
    lastSaveOk = io.save(path, header .. table.concat(rows, '\n') .. '\n')
  end

  local sim = ac.getSim()
  local car = ac.getCar(0)
  local rel = ac.getCameraPositionRelativeToCar()
  ui.text(string.format('replay %s   frame %s / %s   rate %.2f', tostring(sim.isReplayActive),
    tostring(sim.replayCurrentFrame), tostring(sim.replayFrames), sim.replayPlaybackRate))
  ui.text(string.format('camera mode %s   drivable %s   fov %.1f',
    tostring(sim.cameraMode), tostring(sim.driveableCameraMode), sim.cameraFOV))
  ui.text(string.format('camera vs car   x %.3f   y %.3f   z %.3f', rel.x, rel.y, rel.z))
  ui.text(string.format('spline %.4f   rows %d   saved %s', car.splinePosition, #rows,
    lastSaveOk and 'ok' or 'FAILED'))
  if lastError then ui.text('ERROR ' .. lastError) end
end
