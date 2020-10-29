from pgmpy.factors.discrete import DiscreteFactor
import numpy as np
phi1 = DiscreteFactor(["a"], [2], np.random.rand(2))
print(phi1)
print((phi1.values[0])>0.5)