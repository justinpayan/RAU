import pandas as pd
import numpy as np
import pickle

if __name__ == "__main__":
    all_data = []
    for alpha in np.arange(0, 1.01, .1):
        for seed in range(10):
            with open("stat_dict_midl_%d_%.1f.pkl", 'rb') as f:
                x = pickle.load(f)
                opt_usw = x['opt_usw']
                all_data.append([alpha,
                                 seed,
                                 100*x['worst_usw_tpms']/opt_usw,
                                 100*x['worst_usw_maximin']/opt_usw,
                                 100*x['true_usw_tpms']/opt_usw,
                                 100*x['true_usw_maximin']/opt_usw]
                                )
    df = pd.DataFrame(all_data)
    df.columns = [["alpha", "seed", "worst_usw_tpms",
                   "worst_usw_maximin", "true_usw_tpms", "true_usw_maximin"]]
    df.to_csv("stat_df_midl.csv")
