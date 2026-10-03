"""
Annotate and diff the PAG7982 init tables in Halo's open camera driver.

The PixArt-authored driver ships a live "24M" register table and a commented-out
"48M" one, plus a header naming some registers per bank. With no public register
manual, the tables are the best evidence of what each register group does. This
prints each table's final value per (bank, register) with the header's name,
marks the differences, lists registers the 24M table overwrites, and decodes the
multi-byte fields that can be read off directly.

    git clone https://github.com/brilliantlabsAR/halo-firmware
    python tools/halo_sensor/pag7982_regs.py halo-firmware/drivers/video
"""

import argparse
import re
from pathlib import Path

BANK_SELECT = 0xEF
WRITE = re.compile(r"\{0x([0-9A-Fa-f]{2}),\s*0x([0-9A-Fa-f]{2})\}")


def header_names(text):
    """{(bank, register): name} from the `// BANKn` groups in pag7982.h."""
    names, bank = {}, None
    for line in text.splitlines():
        marker = re.match(r"//\s*BANK(\d)", line.strip())
        if marker:
            bank = int(marker.group(1))
            continue
        define = re.match(r"#define\s+(\w+)\s+0x([0-9A-Fa-f]+)", line)
        if define and bank is not None:
            names[(bank, int(define.group(2), 16))] = define.group(1)
    return names


def writes(table):
    """[(bank, register, value)] in order, resolving bank-select writes."""
    out, bank = [], None
    for register, value in WRITE.findall(table):
        register, value = int(register, 16), int(value, 16)
        if register == BANK_SELECT:
            bank = value
        else:
            out.append((bank, register, value))
    return out


def final_values(sequence):
    return {(bank, register): value for bank, register, value in sequence}


def field(values, bank, low, size):
    """Little-endian multi-byte field starting at `low`."""
    return sum(values.get((bank, low + i), 0) << (8 * i) for i in range(size))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("driver_dir", type=Path, help="directory holding pag7982.c and pag7982.h")
    args = parser.parse_args()

    source = (args.driver_dir / "pag7982.c").read_text()
    names = header_names((args.driver_dir / "pag7982.h").read_text())
    table24 = writes(source.split("/* 24M 640 * 480 */")[1].split("};")[0])
    table48 = writes(source.split("// /* 48M 640x480 */")[1].split("// };")[0])
    values24, values48 = final_values(table24), final_values(table48)

    print(f"{'bank':>4} {'reg':>4}  {'name':22s} {'24M':>5} {'48M':>5}")
    for bank, register in sorted(set(values24) | set(values48)):
        v24, v48 = values24.get((bank, register)), values48.get((bank, register))
        show = lambda v: "--" if v is None else f"0x{v:02X}"
        mark = "" if v24 == v48 else "  differs"
        name = names.get((bank, register), "")
        print(f"{bank:>4} 0x{register:02X}  {name:22s} {show(v24):>5} {show(v48):>5}{mark}")

    print("\nOverwritten later in the 24M table (base values, then a 24 MHz patch):")
    seen = {}
    for bank, register, value in table24:
        key = (bank, register)
        if key in seen and seen[key] != value:
            print(f"  bank{bank} 0x{register:02X}: 0x{seen[key]:02X} -> 0x{value:02X}")
        seen[key] = value

    print("\nDecoded fields:")
    for label, values, pclk in (("24M", values24, 24e6), ("48M", values48, 48e6)):
        frame = field(values, 0, 0x4C, 4)
        max_exposure = field(values, 4, 0x48, 4)
        print(
            f"  {label}: R_FRAMETIME {frame} = {2 * frame / pclk * 1e3:.2f} ms ({pclk / (2 * frame):.0f} fps); "
            f"R_AE_MAX_EXPO {max_exposure} (frame - max = {frame - max_exposure}); "
            f"bank2 0x21/0x23 = {field(values, 2, 0x21, 2)} x {field(values, 2, 0x23, 2)}, "
            f"0x25/0x27 = {field(values, 2, 0x25, 2)}, {field(values, 2, 0x27, 2)}; "
            f"AE window {values.get((4, 0x3A), 0) * 4} x {values.get((4, 0x3B), 0) * 4}"
        )

    untouched = [n for k, n in sorted(names.items()) if k not in values24 and k not in values48]
    print("\nNamed in the header but written by neither table (left at reset defaults):")
    print("  " + ", ".join(n for n in untouched if "SKIP" in n or "ISP_WOI" in n))


if __name__ == "__main__":
    main()
