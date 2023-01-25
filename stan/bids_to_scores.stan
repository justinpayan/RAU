data {
  int<lower=0> N; // Number of papers
  int<lower=0> M; // Number of reviewers
  real<lower=0,upper=1> bids[M,N];
}
parameters {
  // Low-rank matrices to approximate scores and bids
  real<lower=0,upper=1> A[M,10]
  real<lower=0,upper=1> B[N,10]

  real<lower=0> sigma_scores[M,N]; // standard deviations for scores
  real<lower=0> sigma_bids[M,N]; // standard deviations for bids
}
transformed parameters {
  real<lower=0,upper=1> mu_bids[M,N] = A*B'; // means for bids
  real<lower=0,upper=1> mu_scores[M,N] = A*B'; // means for scores
}
model {
  // inverse gamma prior on the standard deviations
  // alpha = 3, beta = 0.5 means density is pretty concentrated under 0.5
  sigma_scores ~ inv_gamma(3, 0.5)
  sigma_bids ~ inv_gamma(3, 0.5)

  // draw the bids and scores from their respective normals
  // Note that multivariate normal with diagonal covariance is equivalent to independent, normal vars
  bids ~ normal(mu_bids, sigma_bids)
  scores ~ normal(mu_scores, sigma_scores)
}