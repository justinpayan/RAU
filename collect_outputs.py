import numpy as np
import os

v_tilde_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding/v_tildes"

for query_model in ["random", "tpms", "superstar"]:
    for dset in ["midl", "cvpr", "cvpr18"]:
        expected_usw = []
        true_usw = []
        for seed in range(10):
            with open(os.path.join(v_tilde_dir, "expected_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
                expected_usw.append(float(f.read().strip()))
            with open(os.path.join(v_tilde_dir, "true_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
                true_usw.append(float(f.read().strip()))
        print("Dataset: %s, query_model: %s, mean/std True (Exp): $%.2f \\pm %.2f$ ($%.2f \\pm %.2f$)"
              % (dset, query_model, np.mean(true_usw), np.std(true_usw),
                 np.mean(expected_usw), np.std(expected_usw)))

# for query_model in ["random", "tpms", "superstar"]:
for dset in ["midl", "cvpr", "cvpr18"]:
    expected_usw = []
    true_usw = []
    for seed in range(10):
        with open(os.path.join(v_tilde_dir, "expected_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
            expected_usw.append(float(f.read().strip()))
        with open(os.path.join(v_tilde_dir, "true_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
            true_usw.append(float(f.read().strip()))
    print("Dataset: %s, query_model: %s, mean/std True (Exp): $%.2f \\pm %.2f$ ($%.2f \\pm %.2f$)"
          % (dset, query_model, np.mean(true_usw), np.std(true_usw),
             np.mean(expected_usw), np.std(expected_usw)))
