import numpy as np
import matplotlib.pyplot as plt
import interp

m = interp.load("google/gemma-3-12b-pt")
cap = interp.run(m, "The cat sat on the")
print(cap.tokens)