
# bayesian_model.py
import numpy as np
import jax.numpy as jnp
import jax.scipy.optimize as opt
from scipy.stats import norm
from scipy import linalg
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

def build_integrand(dim, n_preds, scale_prior):
    
    np.random.seed(seed)


    dataset = fetch_openml(name='credit-g', as_frame=True, version=1)
    data = pd.DataFrame(dataset.data)
    target_raw = dataset.target
    target = np.array(target_raw == 'good', dtype=np.float32)
    for col in data.columns:
        if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    preds = data.values.astype(float)
    col_min = preds.min(axis=0)
    col_max = preds.max(axis=0)
    denom = np.where(col_max > col_min, col_max - col_min, 1.0)
    preds = (preds - col_min) / denom


    if n_preds is None:
        n_preds = min(30, preds.shape[0])


    def minuslogpost(beta, _dim, _n_preds):
        data_slice = jnp.array(preds[:_n_preds, :_dim])
        loglik = jnp.sum(-jnp.log(1. + jnp.exp(-jnp.array(target[:_n_preds]) * jnp.dot(data_slice, beta))))
        logprior = -(0.5 / (scale_prior**2)) * jnp.sum(beta**2)
        return -(logprior + loglik)

    beta0 = jnp.zeros(dim)
    rez = opt.minimize(minuslogpost, beta0, args=(dim, n_preds), method='BFGS', options={'maxiter': 100})
    mu = np.array(rez.x, dtype=float)
    Sigma = np.array(rez.hess_inv, dtype=float)

    Cu = linalg.cholesky(Sigma, lower=False)
    log_det_Cu = np.sum(np.log(np.diag(Cu)))

    data_slice_np = preds[:n_preds, :dim]
    target_np = target[:n_preds].astype(float)

    def f(x):
        z = norm.ppf(x)                          
        transformed_beta = mu + Cu @ z            
        log_lik_each = -np.log(1. + np.exp(-target_np * (data_slice_np @ transformed_beta)))
        prior_val = -(0.5 / (scale_prior**2)) * np.sum(transformed_beta**2)
        log_phi_z = np.sum(np.log(norm.pdf(z)))
        # 返回 log：|det C| + sum(loglik) + prior - log phi(z)
        return float(log_det_Cu + np.sum(log_lik_each) + prior_val - log_phi_z)

    return f
