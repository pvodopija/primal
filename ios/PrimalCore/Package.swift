// swift-tools-version: 5.10
// PrimalCore: the GoPro preview receiver (UDP, MPEG-TS, hardware H.264 decode), shared by the
// iPhone app (hotlapp) and the Mac tools in ../tools.
import PackageDescription

let package = Package(
    name: "PrimalCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "PrimalCore", targets: ["PrimalCore"])],
    targets: [.target(name: "PrimalCore")]
)
