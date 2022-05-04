import numpy as np
import torch


# initialize from data
data = [[1, 2], [3, 4]]

tens = torch.tensor(data)

print(f"from list: {tens}")

arr = np.array(data)

tens = torch.from_numpy(arr)


print(torch.tensor([1, 2]))

print(tens)

torch.tensor(data)

tensor = torch.rand((5, 5))

print(f"{tensor.shape=}")
print(f"{tensor.dtype=}")
print(f"{tensor.device=}")

if torch.cuda.is_available():

    arr = [torch.rand((100, 100)).to('cuda') for _ in range(100)]
    print(arr)
    print(*list(map(lambda t: t.device, arr)))

t = arr[0]

















