import numpy as np
import numbersafe as ns

a = ns.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 10]]], 2)
print(a.sangu)