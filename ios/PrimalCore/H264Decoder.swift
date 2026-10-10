import CoreMedia
import Foundation
import VideoToolbox

/// H.264 access units (Annex B) to BGRA pixel buffers, in the hardware decoder (VideoToolbox;
/// on macOS hardware is required, so a stream it cannot take fails here rather than falling
/// back to software). Frames before the first keyframe are dropped.
final class H264Decoder {
    var onFrame: ((CVPixelBuffer, Double) -> Void)?
    private(set) var decoded = 0, failed = 0, beforeKeyframe = 0, hardware = false
    private(set) var width = 0, height = 0
    private var sps: [UInt8]?, pps: [UInt8]?
    private var format: CMVideoFormatDescription?
    private var session: VTDecompressionSession?
    private var keyframeSeen = false

    deinit { if let session { VTDecompressionSessionInvalidate(session) } }

    func decode(_ annexB: [UInt8], pts: Double?) {
        var slices: [[UInt8]] = [], keyframe = false
        for nal in H264Decoder.units(annexB) where !nal.isEmpty {
            switch nal[0] & 0x1F {
            case 7: if sps != nal { sps = nal; format = nil }
            case 8: if pps != nal { pps = nal; format = nil }
            case 9: break  // access unit delimiter
            case let type: keyframe = keyframe || type == 5; slices.append(nal)
            }
        }
        if format == nil { makeSession() }
        guard let session, let format, !slices.isEmpty else { return }
        if !keyframeSeen {
            guard keyframe else { beforeKeyframe += 1; return }
            keyframeSeen = true
        }
        var avcc: [UInt8] = []
        for nal in slices {
            var length = UInt32(nal.count).bigEndian
            withUnsafeBytes(of: &length) { avcc += $0 }
            avcc += nal
        }
        var block: CMBlockBuffer?, sample: CMSampleBuffer?
        guard CMBlockBufferCreateWithMemoryBlock(allocator: kCFAllocatorDefault, memoryBlock: nil, blockLength: avcc.count,
                                                 blockAllocator: kCFAllocatorDefault, customBlockSource: nil, offsetToData: 0,
                                                 dataLength: avcc.count, flags: 0, blockBufferOut: &block) == noErr, let block,
              CMBlockBufferReplaceDataBytes(with: avcc, blockBuffer: block, offsetIntoDestination: 0, dataLength: avcc.count) == noErr
        else { failed += 1; return }
        let time = pts ?? .nan
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: pts.map { CMTime(seconds: $0, preferredTimescale: 90_000) } ?? .invalid,
                                        decodeTimeStamp: .invalid)
        var size = avcc.count
        guard CMSampleBufferCreateReady(allocator: kCFAllocatorDefault, dataBuffer: block, formatDescription: format, sampleCount: 1,
                                        sampleTimingEntryCount: 1, sampleTimingArray: &timing, sampleSizeEntryCount: 1,
                                        sampleSizeArray: &size, sampleBufferOut: &sample) == noErr, let sample
        else { failed += 1; return }
        // no asynchronous flag: the handler runs before this returns, so frames leave in order
        let status = VTDecompressionSessionDecodeFrame(session, sampleBuffer: sample, flags: [], infoFlagsOut: nil) { [weak self] status, _, image, _, _ in
            guard let self else { return }
            if status == noErr, let image { self.decoded += 1; self.onFrame?(image, time) } else { self.failed += 1 }
        }
        if status != noErr { failed += 1 }
    }

    private func makeSession() {
        guard let sps, let pps else { return }
        if let session { VTDecompressionSessionInvalidate(session); self.session = nil }
        let made = sps.withUnsafeBufferPointer { s in
            pps.withUnsafeBufferPointer { p in
                CMVideoFormatDescriptionCreateFromH264ParameterSets(allocator: kCFAllocatorDefault, parameterSetCount: 2,
                                                                    parameterSetPointers: [s.baseAddress!, p.baseAddress!],
                                                                    parameterSetSizes: [s.count, p.count], nalUnitHeaderLength: 4,
                                                                    formatDescriptionOut: &format)
            }
        }
        guard made == noErr, let format else { format = nil; return }
        let dimensions = CMVideoFormatDescriptionGetDimensions(format)
        width = Int(dimensions.width); height = Int(dimensions.height)
        var specification: [CFString: Any] = [:]
        #if os(macOS)
        specification[kVTVideoDecoderSpecification_RequireHardwareAcceleratedVideoDecoder] = true
        #endif
        let attributes: [CFString: Any] = [kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA]
        var created: VTDecompressionSession?
        guard VTDecompressionSessionCreate(allocator: kCFAllocatorDefault, formatDescription: format,
                                           decoderSpecification: specification as CFDictionary,
                                           imageBufferAttributes: attributes as CFDictionary, outputCallback: nil,
                                           decompressionSessionOut: &created) == noErr, let created
        else { self.format = nil; return }
        session = created
        #if os(macOS)
        var usingHardware: CFBoolean?
        _ = withUnsafeMutablePointer(to: &usingHardware) {
            VTSessionCopyProperty(created, key: kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                                  allocator: kCFAllocatorDefault, valueOut: $0)
        }
        hardware = usingHardware == kCFBooleanTrue
        #else
        hardware = true  // iOS decodes H.264 in hardware
        #endif
    }

    /// NAL units of an Annex B byte stream (split at 00 00 01, with or without a leading 00).
    static func units(_ b: [UInt8]) -> [[UInt8]] {
        var out: [[UInt8]] = [], start = -1, i = 0
        while i + 2 < b.count {
            if b[i] == 0 && b[i + 1] == 0 && b[i + 2] == 1 {
                if start >= 0 {
                    var end = i
                    while end > start && b[end - 1] == 0 { end -= 1 }
                    out.append(Array(b[start..<end]))
                }
                i += 3; start = i
            } else {
                i += 1
            }
        }
        if start >= 0 && start < b.count { out.append(Array(b[start...])) }
        return out
    }
}
