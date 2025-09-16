import torch
import numpy as np

input_a = torch.empty(4096, dtype=torch.float32, device="cuda")
input_b = torch.empty(4096, dtype=torch.float32, device="cuda")
output = torch.empty(4096, dtype=torch.float32, device="cuda")


index = 0
flag  = 0 # ADD = 0, SUB = 1
req_id_len = 0

# Take inputs from user

input_a_np = np.fromfile("input.bin", dtype=np.float32, count=4096)
input_a = torch.from_numpy(input_a_np).to("cuda")

flag = int(input("Enter operation (0 for ADD, 1 for SUB): "))
req_id_len = int(input("Enter req_id_len (0-4095): "))

torch.ops.aten.vuln_op(input_a, input_b, output, index, flag, req_id_len)

print("Victim process successfully terminated")
