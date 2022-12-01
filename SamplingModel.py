from cmdstanpy import CmdStanModel

if __name__ == "__main__":
    stan_file = "stan/bernoulli.stan"
    model = CmdStanModel(stan_file=stan_file)
    data_file = "stan/bernoulli.data.json"
    fit = model.sample(data=data_file)
    print(fit.summary())
