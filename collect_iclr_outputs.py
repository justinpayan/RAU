import numpy as np
import os
import pickle
import scipy.sparse

output_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding/outputs"

table_str = "Year & Non-Robust & Robust \\\\\n"

rng = np.random.default_rng(seed=31415)

for year in range(2018, 2023):
    tpms_worst_cases = []
    robust_worst_cases = []
    tpms_avg_cases = []
    robust_avg_cases = []

    for seed in range(10):
        fname = "stat_dict_iclr_%d_%d.pkl" % (year, seed)
        with open(os.path.join(output_dir, fname), 'rb') as f:
            x = pickle.load(f)
            tpms_worst_cases.append(x['worst_usw_tpms'])
            robust_worst_cases.append(x['worst_usw_maxmin'])

        # Get the average case for this
        postfix = "_iclr_%d_%d.npy" % (year, seed)
        mu_fname = "means" + postfix
        mu = np.load(os.path.join(output_dir, mu_fname))

        std_devs_fname = "std_devs" + postfix
        std_devs = np.load(os.path.join(output_dir, std_devs_fname))

        robust_alloc_fname = "max_min_alloc" + postfix
        tpms_alloc_fname = "tpms_alloc" + postfix

        robust_alloc = np.load(os.path.join(output_dir, robust_alloc_fname))
        tpms_alloc = np.load(os.path.join(output_dir, tpms_alloc_fname))

        usw_robust = 0
        usw_tpms = 0
        num_samps = 100
        for i in range(num_samps):
            print(i)
            # Sample from the multivariate normal defined by the means and stds
            draw = rng.multivariate_normal(mu.ravel(), scipy.sparse.diags(std_devs.ravel()).toarray()).reshape(mu.shape)
            draw = np.clip(draw, 0, 1)
            usw_robust += np.sum(robust_alloc * draw)/num_samps
            usw_tpms += np.sum(tpms_alloc * draw)/num_samps
        robust_avg_cases.append(usw_robust)
        tpms_avg_cases.append(usw_tpms)

    table_str += "$%d$ & $%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ & " \
                 "$%.2f \\pm %.2f$ & $%.2f \\pm %.2f$ \\\\\n" % (year,
                                                                 np.mean(tpms_worst_cases),
                                                                 np.std(tpms_worst_cases),
                                                                 np.mean(robust_worst_cases),
                                                                 np.std(robust_worst_cases),
                                                                 np.mean(tpms_avg_cases),
                                                                 np.std(tpms_avg_cases),
                                                                 np.mean(robust_avg_cases),
                                                                 np.std(robust_avg_cases))

print(table_str)
