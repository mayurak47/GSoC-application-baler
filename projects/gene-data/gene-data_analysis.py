
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import scipy.optimize as opt;
import pickle

def analysis(project_name, data_path_before, data_path_after):
    print(data_path_before, data_path_after)
    project_path = f"projects/{project_name}/"

    with open(data_path_before, "rb") as handle:
        before = pickle.load(handle)
        before = before
    with open(data_path_after, "rb") as handle:
        after = pickle.load(handle)

    variable = "logFC_15_s"
    plot_peak(project_path, before[variable], after[variable])

def fit(x, a, b, c, k, m):
     return a*np.exp(-((x-b)**2)/(2*c**2)) + m*x + k

def plot_peak(project_path, before, after):

    x_min = min(before+after)
    x_max = max(before+after)

    with PdfPages(project_path + "/plotting/analysis.pdf") as pdf:    
        
        plt.hist(before, bins=300, label="Before")
        plt.hist(after, bins=300, label="After")
        plt.xlabel("logFC_15_s")
        plt.ylabel("Frequency")
        plt.legend()
        pdf.savefig()
    