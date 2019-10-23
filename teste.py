from utils import utils
import torch

b = torch.tensor([[0, 0, 10, 10],
                  [10,0,20,10]
                 ], dtype=torch.float)
p = torch.tensor([[5, 5, 15, 15],
                  [20, 20, 30, 30]
                  ], dtype=torch.float)

print("jaccard")
a = utils.find_jaccard_overlap(b, p)
print(a)