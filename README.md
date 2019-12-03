# Filter_Response_Norm
A Pytorch implementation of the "Filter Response Normalization Layer"

Simply import it and use it like official batch normalization
```
from FilterResponseNorm import *

input = torch.randn(20, 100, 32, 32)
m = FilterResponseNorm(100)
output = m(input)
```
