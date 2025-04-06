import dualGaraKRNew as dgk
import numpy as np

testram = dgk.dualGaraKRNew(3.4, 33, 61)
print(testram.getTDLs(np.deg2rad(45), np.deg2rad(-45)))