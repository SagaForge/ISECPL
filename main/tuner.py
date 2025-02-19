import numpy as np
import pandas as pd
from tsampler import Sampler
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sampler = Sampler('/home/connor/university/isecpl/temp/LLVM_cleaned.csv', budget=500, inital_sample=0.4, performance_col=None, minimize=True)
    sampler.stage_sampler() ## stage sampler for allocated samples



