import numpy as np
import os
import pickle

output_dir = "/mnt/nfs/scratch1/jpayan/MinimalBidding/outputs"

table_str = "Year & Revs. & Paps. & \\LP{} Adv. & \\algoname{} Adv. & \\LP{} Avg. & \\algoname{} Avg. \\\\\n"

baselines_str = "Year & FairFlow Adv. & PR4A Adv. & FairSeq Adv. & FairFlow Avg. & PR4A Avg. & FairSeq Avg. \\\\\n"

rng = np.random.default_rng(seed=31415)

for year in range(2018, 2023):
    print("Year ", year)
    tpms_worst_cases = []
    robust_worst_cases = []
    tpms_avg_cases = []
    robust_avg_cases = []

    fairflow_worst_cases = []
    pr4a_worst_cases = []
    fairseq_worst_cases = []
    fairflow_avg_cases = []
    pr4a_avg_cases = []
    fairseq_avg_cases = []

    for seed in range(100):
        print("Seed ", seed, flush=True)
        fname = "stat_dict_iclr_%d_%d.pkl" % (year, seed)
        with open(os.path.join(output_dir, fname), 'rb') as f:
            x = pickle.load(f)
            tpms_worst_cases.append(x['worst_usw_tpms'])
            robust_worst_cases.append(x['worst_usw_maxmin'])

        fname = "stat_dict_iclr_baselines_%d_%d.pkl" % (year, seed)
        with open(os.path.join(output_dir, fname), 'rb') as f:
            x = pickle.load(f)
            fairflow_worst_cases.append(x['worst_usw_fairflow'])
            pr4a_worst_cases.append(x['worst_usw_pr4a'])
            fairseq_worst_cases.append(x['worst_usw_fairseq'])

        # Get the average case for this
        postfix = "_iclr_%d_%d.npy" % (year, seed)
        mu_fname = "means" + postfix
        mu = np.load(os.path.join(output_dir, mu_fname))
        m, n = mu.shape

        std_devs_fname = "std_devs" + postfix
        std_devs = np.load(os.path.join(output_dir, std_devs_fname))

        robust_alloc_fname = "max_min_alloc" + postfix
        tpms_alloc_fname = "tpms_alloc" + postfix
        fairflow_alloc_fname = "fairflow_alloc" + postfix
        pr4a_alloc_fname = "pr4a_alloc" + postfix
        fairseq_alloc_fname = "fairseq_alloc" + postfix

        robust_alloc = np.load(os.path.join(output_dir, robust_alloc_fname))
        tpms_alloc = np.load(os.path.join(output_dir, tpms_alloc_fname))
        fairflow_alloc = np.load(os.path.join(output_dir, fairflow_alloc_fname))
        pr4a_alloc = np.load(os.path.join(output_dir, pr4a_alloc_fname))
        fairseq_alloc = np.load(os.path.join(output_dir, fairseq_alloc_fname))

        usw_robust = 0
        usw_tpms = 0
        usw_fairflow = 0
        usw_pr4a = 0
        usw_fairseq = 0
        num_samps = 100
        for i in range(num_samps):
            print(i)
            # Sample from the multivariate normal defined by the means and stds
            draw = rng.normal(loc=mu, scale=std_devs)
            draw = np.clip(draw, 0, 1)
            usw_robust += np.sum(robust_alloc * draw) / num_samps
            usw_tpms += np.sum(tpms_alloc * draw) / num_samps
            usw_fairflow += np.sum(fairflow_alloc * draw) / num_samps
            usw_pr4a += np.sum(pr4a_alloc * draw) / num_samps
            usw_fairseq += np.sum(fairseq_alloc * draw) / num_samps
        robust_avg_cases.append(usw_robust)
        tpms_avg_cases.append(usw_tpms)
        fairflow_avg_cases.append(usw_fairflow)
        pr4a_avg_cases.append(usw_pr4a)
        fairseq_avg_cases.append(usw_fairseq)

    table_str += "$%d$ & $%d$ & $%d$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & " \
                 "$%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\\n" % (year,
                                                                 m,
                                                                 n,
                                                                 np.mean(tpms_worst_cases) / n,
                                                                 np.std(tpms_worst_cases) / n,
                                                                 np.mean(robust_worst_cases) / n,
                                                                 np.std(robust_worst_cases) / n,
                                                                 np.mean(tpms_avg_cases) / n,
                                                                 np.std(tpms_avg_cases) / n,
                                                                 np.mean(robust_avg_cases) / n,
                                                                 np.std(robust_avg_cases) / n)
    baselines_str += "$%d$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & " \
                     "$%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\\n" % (year,
                                                                     np.mean(fairflow_worst_cases) / n,
                                                                     np.std(fairflow_worst_cases) / n,
                                                                     np.mean(pr4a_worst_cases) / n,
                                                                     np.std(pr4a_worst_cases) / n,
                                                                     np.mean(fairseq_worst_cases) / n,
                                                                     np.std(fairseq_worst_cases) / n,
                                                                     np.mean(fairflow_avg_cases) / n,
                                                                     np.std(fairflow_avg_cases) / n,
                                                                     np.mean(pr4a_avg_cases) / n,
                                                                     np.std(pr4a_avg_cases) / n,
                                                                     np.mean(fairseq_avg_cases) / n,
                                                                     np.std(fairseq_avg_cases) / n)

    print(table_str)
    print(baselines_str)
