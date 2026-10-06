// Text in images with macOS's Vision framework, no install: image paths on stdin, one per
// line; prints "path<TAB>text" per image, the recognised lines joined by " | ".
// Usage: ls crops/*.png | xcrun swift experiments/tools/vision_ocr.swift
import Foundation
import Vision
import AppKit

while let path = readLine() {
    guard let image = NSImage(contentsOfFile: path),
          let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        print("\(path)\t"); continue
    }
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    try? VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
    let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
    print("\(path)\t\(lines.joined(separator: " | "))")
    fflush(stdout)
}
