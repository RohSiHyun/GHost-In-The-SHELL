import re
from pathlib import Path
def parse_sass(filename):
    instrs = []

    with open(filename, "r") as f:
        lines = f.readlines()

    hex_re = re.compile(r"/\*\s*0x([0-9a-fA-F]+)\s*\*/")

    current = []
    for line in lines:
        m = hex_re.findall(line)
        if m:
            val = int(m[0], 16)
            b = val.to_bytes(8, byteorder="little")
            current.append(b)

            if len(current) == 2:
                instrs.append(b"".join(current))
                current = []

    return instrs

instr_bytes = parse_sass("payload.sass")

for i, inst in enumerate(instr_bytes):
    print(f"Instr {i:04d}: {inst.hex()}")

binary_blob = b"".join(instr_bytes)
cu_count = len(binary_blob)
print("CU object byte count: ", cu_count)

padding_size = 2048 - cu_count
binary_blob += b"\x00" * padding_size


with open("input.bin", "wb") as f:
    f.write(binary_blob)

print("Total payload size (SASS + C code): ", len(binary_blob))
