import Darwin
import Foundation

/// The GoPro's Wi-Fi preview (HERO4-HERO8, gpControl): H.264 in MPEG-TS over UDP, sent to
/// our port 8554 while we send the camera a keep-alive datagram every ~2.5 s. The stream
/// itself is started over HTTP (`/gp/gpControl/execute?p1=gpStream&a1=proto_v2&c1=restart`).
final class GoProPreviewSocket {
    static let keepAliveInterval = 2.5
    let fd: Int32
    private var camera = sockaddr_in()

    init(cameraIP: String = "10.5.5.9", port: UInt16 = 8554) throws {
        fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)
        guard fd >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        var yes: Int32 = 1, buffer: Int32 = 4 << 20
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buffer, socklen_t(MemoryLayout<Int32>.size))
        var timeout = timeval(tv_sec: 0, tv_usec: 500_000)
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))
        var local = sockaddr_in()
        local.sin_family = sa_family_t(AF_INET)
        local.sin_port = port.bigEndian
        let bound = withUnsafePointer(to: &local) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
        }
        guard bound == 0 else { close(fd); throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EADDRINUSE) }
        camera.sin_family = sa_family_t(AF_INET)
        camera.sin_port = port.bigEndian
        inet_pton(AF_INET, cameraIP, &camera.sin_addr)
    }

    deinit { close(fd) }

    func keepAlive() {
        let message = Array("_GPHD_:0:0:2:0.000000\n".utf8)
        _ = withUnsafePointer(to: &camera) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                sendto(fd, message, message.count, 0, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
    }

    /// One datagram into `buffer`; its length, or 0 on the receive timeout.
    func receive(into buffer: inout [UInt8]) -> Int {
        max(recv(fd, &buffer, buffer.count, 0), 0)
    }
}

/// MPEG-TS packets to H.264 access units (Annex B bytes) with their presentation time, s.
final class TSDemuxer {
    var onAccessUnit: (([UInt8], Double?) -> Void)?
    private(set) var packets = 0, continuityErrors = 0, accessUnits = 0
    private var pmtPID: Int?, videoPID: Int?
    private var pes: [UInt8] = [], pesPTS: Double?, pesExpected = 0
    private var lastCC = -1

    func feed(_ data: UnsafeBufferPointer<UInt8>) {
        var i = 0
        while i + 188 <= data.count {
            if data[i] == 0x47 { packet(data, i); i += 188 } else { i += 1 }
        }
    }

    private func packet(_ d: UnsafeBufferPointer<UInt8>, _ o: Int) {
        packets += 1
        let start = d[o + 1] & 0x40 != 0
        let pid = Int(d[o + 1] & 0x1F) << 8 | Int(d[o + 2])
        let control = (d[o + 3] >> 4) & 0x3, cc = Int(d[o + 3] & 0xF)
        var p = o + 4
        if control & 0x2 != 0 { p += 1 + Int(d[p]) }
        guard control & 0x1 != 0, p < o + 188 else { return }
        let payload = Array(d[p..<(o + 188)])
        if pid == 0 {
            if start { parsePAT(payload) }
        } else if pid == pmtPID {
            if start { parsePMT(payload) }
        } else if pid == videoPID {
            if lastCC >= 0 && cc != (lastCC + 1) & 0xF { continuityErrors += 1 }
            lastCC = cc
            if start {
                flush()
                guard payload.count >= 9, payload[0] == 0, payload[1] == 0, payload[2] == 1 else { return }
                let length = Int(payload[4]) << 8 | Int(payload[5]), headerLength = Int(payload[8])
                guard payload.count >= 9 + headerLength else { return }
                if payload[7] & 0x80 != 0 { pesPTS = TSDemuxer.timestamp(payload, 9) }
                pesExpected = length > 0 ? length - 3 - headerLength : 0
                pes = Array(payload[(9 + headerLength)...])
            } else if !pes.isEmpty {
                pes += payload
            }
            // a bounded PES is complete without waiting for the next one: a frame sooner
            if pesExpected > 0 && pes.count >= pesExpected { flush() }
        }
    }

    private func flush() {
        guard !pes.isEmpty else { return }
        accessUnits += 1
        onAccessUnit?(pes, pesPTS)
        pes = []; pesPTS = nil; pesExpected = 0
    }

    private static func timestamp(_ b: [UInt8], _ i: Int) -> Double {
        let ticks = (UInt64(b[i] >> 1) & 0x7) << 30 | UInt64(b[i + 1]) << 22 | UInt64(b[i + 2] >> 1) << 15
            | UInt64(b[i + 3]) << 7 | UInt64(b[i + 4] >> 1)
        return Double(ticks) / 90_000
    }

    private static func section(_ payload: [UInt8]) -> ArraySlice<UInt8>? {
        let s = 1 + Int(payload[0])
        guard payload.count >= s + 3 else { return nil }
        let length = Int(payload[s + 1] & 0x0F) << 8 | Int(payload[s + 2])
        let end = min(s + 3 + length - 4, payload.count)  // the CRC out
        return end > s ? payload[s..<end] : nil
    }

    private func parsePAT(_ payload: [UInt8]) {
        guard let t = TSDemuxer.section(payload) else { return }
        var i = t.startIndex + 8
        while i + 4 <= t.endIndex {
            let program = Int(t[i]) << 8 | Int(t[i + 1]), pid = Int(t[i + 2] & 0x1F) << 8 | Int(t[i + 3])
            if program != 0 { pmtPID = pid; return }
            i += 4
        }
    }

    private func parsePMT(_ payload: [UInt8]) {
        guard let t = TSDemuxer.section(payload), t.count >= 12 else { return }
        let s = t.startIndex
        var i = s + 12 + (Int(t[s + 10] & 0x0F) << 8 | Int(t[s + 11]))
        while i + 5 <= t.endIndex {
            let type = t[i], pid = Int(t[i + 1] & 0x1F) << 8 | Int(t[i + 2])
            if type == 0x1B { videoPID = pid; return }  // H.264
            i += 5 + (Int(t[i + 3] & 0x0F) << 8 | Int(t[i + 4]))
        }
    }
}
