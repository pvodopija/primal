// The GoPro preview through ios/PrimalCore on the Mac: receives, unpacks and decodes in
// hardware, saves the raw stream, and writes each frame to stdout for
// experiments/gopro_preview.py:
//   "PRMF", pts s (f64), arrival unix s (f64), decode ms (f64), width (u32), height (u32), BGRA rows.
// Statistics go to stderr every 5 s.
//
// While the camera records, the HERO7's preview runs at the recording's frame rate (60 fps
// in the second run) where PRIMAL needs 30: every frame is decoded (each depends on the last)
// but at most --max-fps are written out, by their timestamps.
//
// Usage: gopro_probe [--ts stream.ts] [--seconds N] [--camera 10.5.5.9] [--max-fps 30]
import CoreVideo
import Foundation

signal(SIGPIPE, SIG_IGN)
var options: [String: String] = [:]
var argv = CommandLine.arguments.dropFirst().makeIterator()
while let key = argv.next(), let value = argv.next() { options[key] = value }
let seconds = Double(options["--seconds"] ?? "") ?? .infinity
var tsOut: FileHandle?
if let path = options["--ts"] {
    FileManager.default.createFile(atPath: path, contents: nil)
    tsOut = FileHandle(forWritingAtPath: path)
}

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }

func writeAll(_ p: UnsafeRawPointer, _ n: Int) {
    var done = 0
    while done < n {
        let w = write(1, p + done, n - done)
        if w <= 0 { log("gopro_probe: the reader closed; stopping"); exit(0) }
        done += w
    }
}

let preview: GoProPreviewSocket
do { preview = try GoProPreviewSocket(cameraIP: options["--camera"] ?? "10.5.5.9") } catch { log("gopro_probe: no socket on port 8554: \(error)"); exit(1) }
let demuxer = TSDemuxer(), decoder = H264Decoder()
var decodeStarted = 0.0, lastOut = -Double.infinity, lastPTS = -Double.infinity, skipped = 0, reordered = false
let minSpacing = 1 / (Double(options["--max-fps"] ?? "30") ?? 30) - 0.004

decoder.onFrame = { image, pts in
    // too soon after the last one written: skip it (a jump back is a timestamp restart: keep it).
    // A stream whose frames come out of order (B-frames: videos fake_gopro replays, not the
    // GoPro's preview) is passed on whole.
    if pts.isFinite && pts < lastPTS && pts > lastPTS - 0.5 { reordered = true }
    if pts.isFinite { lastPTS = pts }
    if !reordered && pts.isFinite && pts > lastOut - 0.5 && pts - lastOut < minSpacing { skipped += 1; return }
    if pts.isFinite { lastOut = pts }
    let decodeMs = (Date().timeIntervalSince1970 - decodeStarted) * 1000
    CVPixelBufferLockBaseAddress(image, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    let w = CVPixelBufferGetWidth(image), h = CVPixelBufferGetHeight(image), stride = CVPixelBufferGetBytesPerRow(image)
    guard let base = CVPixelBufferGetBaseAddress(image) else { return }
    var header = Data("PRMF".utf8)
    for v in [pts, Date().timeIntervalSince1970, decodeMs] { withUnsafeBytes(of: v) { header.append(contentsOf: $0) } }
    for v in [UInt32(w), UInt32(h)] { withUnsafeBytes(of: v) { header.append(contentsOf: $0) } }
    header.withUnsafeBytes { writeAll($0.baseAddress!, $0.count) }
    for y in 0..<h { writeAll(base + y * stride, w * 4) }
}
demuxer.onAccessUnit = { unit, pts in
    decodeStarted = Date().timeIntervalSince1970
    decoder.decode(unit, pts: pts)
}

var buffer = [UInt8](repeating: 0, count: 65_536)
let start = Date().timeIntervalSince1970
var lastKeepAlive = -100.0, lastStats = start, datagrams = 0, bytes = 0, bytesAtStats = 0
preview.keepAlive()
log("gopro_probe: listening on UDP 8554, keep-alive to \(options["--camera"] ?? "10.5.5.9")")
while Date().timeIntervalSince1970 - start < seconds {
    let now = Date().timeIntervalSince1970
    if now - lastKeepAlive >= GoProPreviewSocket.keepAliveInterval { preview.keepAlive(); lastKeepAlive = now }
    let n = preview.receive(into: &buffer)
    if n > 0 {
        datagrams += 1; bytes += n
        tsOut?.write(Data(buffer[0..<n]))
        buffer.withUnsafeBufferPointer { demuxer.feedDatagram(UnsafeBufferPointer(rebasing: $0[0..<n])) }
    }
    if now - lastStats >= 5 {
        log(String(format: "gopro_probe: %.0f s  %d datagrams  %.0f kbit/s  %d TS packets (%d continuity errors)  %d frames in, %d decoded %dx%d (%@), %d failed, %d before the first keyframe, %d not passed on (over the frame rate)",
                   now - start, datagrams, Double(bytes - bytesAtStats) * 8 / 1000 / (now - lastStats), demuxer.packets, demuxer.continuityErrors,
                   demuxer.accessUnits, decoder.decoded, decoder.width, decoder.height, decoder.hardware ? "hardware" : "NOT hardware",
                   decoder.failed, decoder.beforeKeyframe, skipped))
        lastStats = now; bytesAtStats = bytes
    }
}
tsOut?.closeFile()
