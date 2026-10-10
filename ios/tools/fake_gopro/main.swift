// A stand-in for the GoPro's Wi-Fi preview, to test the receiving side without the camera:
// replays an H.264 video file as MPEG-TS over UDP to 127.0.0.1:8554 in real time, PAT and
// PMT every half second, the parameter sets before every keyframe, 7 TS packets a datagram
// (the section CRCs are left zero; our demuxer does not read them).
//
// A recorded preview (a .ts that gopro_probe saved) is sent as it came, paced by its video
// timestamps (restarts included).
//
// Usage: fake_gopro VIDEO.mp4|STREAM.ts [--seconds N] [--start S] [--port 8554]
import AVFoundation
import Darwin

var options: [String: String] = [:]
var argv = CommandLine.arguments.dropFirst(2).makeIterator()
while let key = argv.next(), let value = argv.next() { options[key] = value }
let path = CommandLine.arguments[1]
let seconds = Double(options["--seconds"] ?? "") ?? .infinity, startAt = Double(options["--start"] ?? "0") ?? 0

let fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
var to = sockaddr_in()
to.sin_family = sa_family_t(AF_INET)
to.sin_port = UInt16(options["--port"] ?? "8554")!.bigEndian
inet_pton(AF_INET, "127.0.0.1", &to.sin_addr)

if path.hasSuffix(".ts") {
    let data = try! Data(contentsOf: URL(fileURLWithPath: path))
    let demuxer = TSDemuxer()
    var clock = 0.0, lastPTS: Double?, chunkTime = 0.0
    demuxer.onAccessUnit = { _, pts in
        guard let pts else { return }
        if let last = lastPTS { clock += (pts > last && pts - last < 1) ? pts - last : 1 / 60 }
        lastPTS = pts; chunkTime = clock
    }
    let start = Date().timeIntervalSince1970
    // the saved stream is the camera's datagrams back to back: a header, then TS slots up to the next header
    let bytes = [UInt8](data)
    func isHeader(_ o: Int) -> Bool {
        guard o + 12 < bytes.count, bytes[o] & 0x80 != 0, bytes[o + 12] == 0x47 else { return false }
        let length = Int(bytes[o + 10]) << 8 | Int(bytes[o + 11])
        return length > 0 && length <= 7 * 188 && length % 188 == 0
    }
    var o = 0
    while o < bytes.count && chunkTime < seconds {
        var end = o + 12 + 188
        while end < bytes.count && !isHeader(end) { end += 188 }
        let datagram = Array(bytes[o..<min(end, bytes.count)])
        datagram.withUnsafeBufferPointer { demuxer.feedDatagram($0) }
        let wait = start + chunkTime - Date().timeIntervalSince1970
        if wait > 0 { usleep(useconds_t(wait * 1e6)) }
        _ = withUnsafePointer(to: &to) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { sendto(fd, datagram, datagram.count, 0, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        o = end
    }
    FileHandle.standardError.write("fake_gopro: replayed \(String(format: "%.0f", chunkTime)) s of \(path)\n".data(using: .utf8)!)
    exit(0)
}

let asset = AVURLAsset(url: URL(fileURLWithPath: path))
let reader = try! AVAssetReader(asset: asset)
let track = asset.tracks(withMediaType: .video)[0]
reader.timeRange = CMTimeRange(start: CMTime(seconds: startAt, preferredTimescale: 600), duration: .positiveInfinity)
let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)  // compressed samples as stored
reader.add(output)
reader.startReading()

var counters: [Int: Int] = [:], pending: [UInt8] = []
func send(_ packet: [UInt8]) {
    pending += packet
    guard pending.count >= 7 * 188 else { return }
    _ = withUnsafePointer(to: &to) {
        $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { sendto(fd, pending, pending.count, 0, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    pending = []
}

/// One PES or section split into 188-byte TS packets, the last padded by adaptation stuffing.
func packetize(_ pid: Int, _ payload: [UInt8]) {
    var i = 0
    while i < payload.count {
        let chunk = min(184, payload.count - i), cc = counters[pid, default: 0]
        counters[pid] = (cc + 1) & 0xF
        var p: [UInt8] = [0x47, UInt8((i == 0 ? 0x40 : 0) | (pid >> 8)), UInt8(pid & 0xFF)]
        if chunk < 184 {
            let fill = 184 - chunk  // adaptation field: its length byte, flags, stuffing
            p.append(UInt8(0x30 | cc)); p.append(UInt8(fill - 1))
            if fill > 1 { p.append(0); p += [UInt8](repeating: 0xFF, count: fill - 2) }
        } else {
            p.append(UInt8(0x10 | cc))
        }
        p += payload[i..<(i + chunk)]
        send(p); i += chunk
    }
}

func tables() {
    packetize(0, [0, 0x00, 0xB0, 13, 0, 1, 0xC1, 0, 0, 0, 1, 0xE1, 0x00, 0, 0, 0, 0])  // program 1 -> PMT PID 0x100
    packetize(0x100, [0, 0x02, 0xB0, 18, 0, 1, 0xC1, 0, 0, 0xE1, 0x01, 0xF0, 0, 0x1B, 0xE1, 0x01, 0xF0, 0, 0, 0, 0, 0])  // H.264 on PID 0x101
}

let start = Date().timeIntervalSince1970
var firstPTS: Double?, lastTables = -1.0, frames = 0
let startCode: [UInt8] = [0, 0, 0, 1]
while let sample = output.copyNextSampleBuffer() {
    guard let block = CMSampleBufferGetDataBuffer(sample), let format = CMSampleBufferGetFormatDescription(sample) else { continue }
    let pts = CMSampleBufferGetPresentationTimeStamp(sample).seconds
    if firstPTS == nil { firstPTS = pts }
    let t = pts - firstPTS!
    if t > seconds { break }
    let wait = start + t - Date().timeIntervalSince1970
    if wait > 0 { usleep(useconds_t(wait * 1e6)) }
    if t - lastTables >= 0.5 { tables(); lastTables = t }
    var length = 0, data: UnsafeMutablePointer<CChar>?
    CMBlockBufferGetDataPointer(block, atOffset: 0, lengthAtOffsetOut: nil, totalLengthOut: &length, dataPointerOut: &data)
    let avcc = UnsafeRawBufferPointer(start: data, count: length)
    var nals: [[UInt8]] = [], i = 0, keyframe = false
    while i + 4 <= length {
        let n = Int(avcc[i]) << 24 | Int(avcc[i + 1]) << 16 | Int(avcc[i + 2]) << 8 | Int(avcc[i + 3])
        let nal = Array(avcc[(i + 4)..<min(i + 4 + n, length)])
        keyframe = keyframe || (nal.first ?? 0) & 0x1F == 5
        nals.append(nal); i += 4 + n
    }
    var es: [UInt8] = startCode + [0x09, 0xF0]  // access unit delimiter
    if keyframe {
        for index in 0..<2 {
            var p: UnsafePointer<UInt8>?, size = 0
            CMVideoFormatDescriptionGetH264ParameterSetAtIndex(format, parameterSetIndex: index, parameterSetPointerOut: &p,
                                                               parameterSetSizeOut: &size, parameterSetCountOut: nil, nalUnitHeaderLengthOut: nil)
            es += startCode + Array(UnsafeBufferPointer(start: p, count: size))
        }
    }
    for nal in nals { es += startCode + nal }
    let ticks = UInt64(max(t, 0) * 90_000) + 90_000
    let ptsBytes: [UInt8] = [UInt8(0x21 | ((ticks >> 29) & 0x0E)), UInt8((ticks >> 22) & 0xFF), UInt8(((ticks >> 14) & 0xFE) | 1),
                             UInt8((ticks >> 7) & 0xFF), UInt8(((ticks << 1) & 0xFE) | 1)]
    packetize(0x101, [0, 0, 1, 0xE0, 0, 0, 0x80, 0x80, 5] + ptsBytes + es)  // unbounded video PES
    frames += 1
}
FileHandle.standardError.write("fake_gopro: sent \(frames) frames\n".data(using: .utf8)!)
