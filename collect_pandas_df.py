import pandas as pd
import numpy as np
import pickle
import os

# if __name__ == "__main__":
#     all_data = []
#     for alpha in np.arange(0, 1.01, .1):
#         for seed in range(10):
#             fname = "slurm/stat_dict_midl_%d_%.1f.pkl" % (seed, alpha)
#             if os.path.isfile(fname):
#                 with open(fname, 'rb') as f:
#                     x = pickle.load(f)
#                     opt_usw = x['opt_usw']
#                     all_data.append([alpha,
#                                      seed,
#                                      100*x['worst_usw_tpms']/opt_usw,
#                                      100*x['worst_usw_maxmin']/opt_usw,
#                                      100*x['true_usw_tpms']/opt_usw,
#                                      100*x['true_usw_maxmin']/opt_usw]
#                                     )
#     df = pd.DataFrame(all_data)
#     df.columns = [["alpha", "seed", "worst_usw_tpms",
#                    "worst_usw_maxmin", "true_usw_tpms", "true_usw_maxmin"]]
#     df.to_csv("stat_df_midl.csv", index=False)

# if __name__ == "__main__":
#     all_data = []
#     for num_dummies in range(0, 101, 5):
#         for seed in range(100):
#             fname = "outputs/stat_dict_dummy_revs_midl_%d_%d.pkl" % (num_dummies, seed)
#             if os.path.isfile(fname):
#                 with open(fname, 'rb') as f:
#                     x = pickle.load(f)
#                     opt_usw = x['opt_usw']
#                     all_data.append([num_dummies,
#                                      seed,
#                                      'Naive LP',
#                                      100*x['true_usw_tpms']/opt_usw]
#                                     )
#                     all_data.append([num_dummies,
#                                      seed,
#                                      'RRA',
#                                      100 * x['true_usw_maxmin'] / opt_usw]
#                                     )
#     df = pd.DataFrame(all_data)
#     df.columns = [["num_dummies", "seed", "Solver", "true_value"]]
#     df.to_csv("stat_df_midl_dummy_revs.csv", index=False)


if __name__ == "__main__":
    all_data = []
    for num_dummies in range(0, 101, 5):
        for seed in range(100):
            fname = "outputs/stat_dict_dummy_paps_midl_%d_%d.pkl" % (num_dummies, seed)
            if os.path.isfile(fname):
                with open(fname, 'rb') as f:
                    x = pickle.load(f)
                    opt_usw = x['opt_usw']
                    opt_usw_dummies = x['opt_obj_dummies']
                    all_data.append([num_dummies,
                                     seed,
                                     'Naive LP',
                                     False,
                                     100*x['true_usw_tpms']/opt_usw]
                                    )
                    all_data.append([num_dummies,
                                     seed,
                                     'RRA',
                                     False,
                                     100 * x['true_usw_maxmin'] / opt_usw]
                                    )
                    all_data.append([num_dummies,
                                     seed,
                                     'Naive LP',
                                     True,
                                     100 * x['true_obj_dummies'] / opt_usw_dummies]
                                    )
                    all_data.append([num_dummies,
                                     seed,
                                     'RRA',
                                     True,
                                     100 * x['true_obj_max_min_dummies'] / opt_usw_dummies]
                                    )
    df = pd.DataFrame(all_data)
    df.columns = [["num_dummies", "seed", "Solver", "DummiesOnly", "true_value"]]
    df.to_csv("stat_df_midl_dummy_paps.csv", index=False)
