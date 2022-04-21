import numpy as np
import os

data_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding"
v_tilde_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding/v_tildes"

expected_usw = {}
init_true_usw = {}
opt_usw = {}
dsets = ["midl", "cvpr", "cvpr18"]
qms = ["random", "tpms", "superstar", "randomsample"]

for dset in dsets:
    expected_usw_list = []
    init_true_usw_list = []
    opt_usw_list = []
    for seed in range(10):
        expected_usw_list.append(np.load(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d.npy" % (dset, seed))))
        init_true_usw_list.append(np.load(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d_true.npy" % (dset, seed))))
        opt_usw_list.append(np.load(os.path.join(data_dir, "saved_init_expected_usw", "%s_%d_opt.npy" % (dset, seed))))

    # print("Dataset: %s, mean/std True (Exp): $%.2f \\pm %.2f$ ($%.2f \\pm %.2f$), OPT: $%.2f \\pm %.2f$"
    #       % (dset, np.mean(init_true_usw_list), np.std(init_true_usw_list),
    #          np.mean(expected_usw_list), np.std(expected_usw_list), np.mean(opt_usw_list), np.std(opt_usw_list)))
    init_true_usw[dset] = init_true_usw_list
    opt_usw[dset] = opt_usw_list
    expected_usw[dset] = expected_usw_list

print("No queries & $%.2f \\pm %.2f$ ($%.2f \\pm %.2f$) & "
      "$%.2f \\pm %.2f$ ($%.2f \\pm %.2f$) & "
      "$%.2f \\pm %.2f$ ($%.2f \\pm %.2f$) \\\\" % (np.mean(init_true_usw["midl"]), np.std(init_true_usw["midl"]),
                                                    np.mean(expected_usw["midl"]), np.std(expected_usw["midl"]),
                                                    np.mean(init_true_usw["cvpr"]), np.std(init_true_usw["cvpr"]),
                                                    np.mean(expected_usw["cvpr"]), np.std(expected_usw["cvpr"]),
                                                    np.mean(init_true_usw["cvpr18"]), np.std(init_true_usw["cvpr18"]),
                                                    np.mean(expected_usw["cvpr18"]), np.std(expected_usw["cvpr18"]),
                                                    ))


str_map = {'random': 'Random', 'tpms': "TPMS", 'superstar': 'SUPER*',
           'supergreedymax': "SUPER* + Max Exp. Value", "randomsample": "Random Sample"}
for query_model in qms:
    table_str = str_map[query_model] + " "
    for dset in dsets:
        expected_usw = []
        true_usw = []
        for seed in range(10):
            with open(os.path.join(v_tilde_dir, "expected_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
                eusw = float(f.read().strip())
                expected_usw.append(eusw)
            with open(os.path.join(v_tilde_dir, "true_obj_%s_%s_%d" % (dset, query_model, seed)), 'r') as f:
                tusw = float(f.read().strip())
                progress = 100.0*(tusw - init_true_usw[dset][seed])/(opt_usw[dset][seed] - init_true_usw[dset][seed])
                true_usw.append(progress)
        table_str += "& $%.2f \\pm %.2f$ \\%% ($%.2f \\pm %.2f$) " % \
                     (np.mean(true_usw), np.std(true_usw), np.mean(expected_usw), np.std(expected_usw))
    table_str += "\\\\"
    print(table_str)
        # print("Dataset: %s, query_model: %s, mean/std True Progress (Exp): $%.2f\\%% \\pm %.2f\\%%$ ($%.2f \\pm %.2f$)"
        #       % (dset, query_model, np.mean(true_usw), np.std(true_usw),
        #          np.mean(expected_usw), np.std(expected_usw)))

print("OPT & $%.2f \\pm %.2f$ (NA) & "
      "$%.2f \\pm %.2f$ (NA) & "
      "$%.2f \\pm %.2f$ (NA) \\\\" % (np.mean(opt_usw["midl"]), np.std(opt_usw["midl"]),
                                        np.mean(opt_usw["cvpr"]), np.std(opt_usw["cvpr"]),
                                        np.mean(opt_usw["cvpr18"]), np.std(opt_usw["cvpr18"]),
                                        ))
