import numpy as np
import os

data_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding"
v_tilde_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding/v_tildes"

# for query_model in ["random", "tpms", "superstar"]:
#     for dset in ["midl", "cvpr", "cvpr18"]:
#         expected_usw = []
#         true_usw = []
#         for seed in range(10):
#             with open(os.path.join(v_tilde_dir, "expected_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
#                 expected_usw.append(float(f.read().strip()))
#             with open(os.path.join(v_tilde_dir, "true_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
#                 true_usw.append(float(f.read().strip()))
#         print("Dataset: %s, query_model: %s, mean/std True (Exp): $%.2f \\pm %.2f$ ($%.2f \\pm %.2f$)"
#               % (dset, query_model, np.mean(true_usw), np.std(true_usw),
#                  np.mean(expected_usw), np.std(expected_usw)))

# for query_model in ["random", "tpms", "superstar"]:
for dset in ["midl", "cvpr", "cvpr18"]:
    expected_usw = []
    true_usw = []
    opt_usw = []
    for seed in range(10):
        expected_usw.append(np.load(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d.npy" % (dset, seed))))
        true_usw.append(np.load(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d_true.npy" % (dset, seed))))
        opt_usw.append(np.load(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d_opt.npy" % (dset, seed))))

    print("Dataset: %s, mean/std True (Exp): $%.2f \\pm %.2f$ ($%.2f \\pm %.2f$), OPT: $%.2f \\pm %.2f$"
          % (dset, np.mean(true_usw), np.std(true_usw),
             np.mean(expected_usw), np.std(expected_usw), np.mean(opt_usw), np.std(opt_usw)))
