import numpy as np
import os
import pickle

output_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding/outputs"

table_str = "Year & Non-Robust & Robust \\\\\n"

for year in range(2018, 2023):
    tpms_worst_cases = []
    robust_worst_cases = []

    for seed in range(10):
        fname = "stat_dict_iclr_%d_%d.pkl" % (year, seed)
        with open(os.path.join(output_dir, fname), 'rb') as f:
            x = pickle.load(f)
            tpms_worst_cases.append(x['worst_usw_tpms'])
            robust_worst_cases.append(x['worst_usw_maxmin'])

    table_str += "%d & %.2f \\pm %.2f & %.2f \\pm %.2f \\\\\n" % (year,
                                                                  np.mean(tpms_worst_cases),
                                                                  np.std(tpms_worst_cases),
                                                                  np.mean(robust_worst_cases),
                                                                  np.std(robust_worst_cases))

print(table_str)