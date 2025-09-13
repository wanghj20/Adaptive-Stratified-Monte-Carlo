
# bayesian_model.py
import numpy as np
import jax.numpy as jnp
import jax.scipy.optimize as opt
from scipy.stats import norm
from scipy import linalg
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml

def build_integrand(dim=5, n_preds=None, scale_prior=1.0, seed=15):
    """
    构建你的被积函数 f(x)（返回 log-值），以及使用的维度 dim。
    步骤与原脚本一致：拉取 OpenML credit-g，预处理，做 Laplace (MAP+BFGS) 得 mu, Sigma，再定义 f(x)。

    返回：
      f: callable, 输入 x∈[0,1]^dim，输出 log-值（float）
      dim: 使用的维度（与入参一致）
    """
    np.random.seed(seed)

    # 1) 数据集
    dataset = fetch_openml(name='credit-g', as_frame=True, version=1)
    data = pd.DataFrame(dataset.data)
    target_raw = dataset.target

    # 2) 目标：good -> 1.0, 其他 -> 0.0
    target = np.array(target_raw == 'good', dtype=np.float32)

    # 3) 类别列编码
    for col in data.columns:
        if data[col].dtype == 'object' or str(data[col].dtype).startswith('category'):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

    # 4) numpy + 列级 min-max 归一
    preds = data.values.astype(float)
    col_min = preds.min(axis=0)
    col_max = preds.max(axis=0)
    denom = np.where(col_max > col_min, col_max - col_min, 1.0)
    preds = (preds - col_min) / denom

    # 5) 选用的样本行数
    if n_preds is None:
        n_preds = min(30, preds.shape[0])

    # 6) 负对数后验（保持你原有的 0/1 写法——注意这其实对应 ±1 标签形式，但按你的原式不改动）
    def minuslogpost(beta, _dim, _n_preds):
        data_slice = jnp.array(preds[:_n_preds, :_dim])
        loglik = jnp.sum(-jnp.log(1. + jnp.exp(-jnp.array(target[:_n_preds]) * jnp.dot(data_slice, beta))))
        logprior = -(0.5 / (scale_prior**2)) * jnp.sum(beta**2)
        return -(logprior + loglik)

    # 7) 用 BFGS 做 MAP
    beta0 = jnp.zeros(dim)
    rez = opt.minimize(minuslogpost, beta0, args=(dim, n_preds), method='BFGS', options={'maxiter': 100})
    mu = np.array(rez.x, dtype=float)
    Sigma = np.array(rez.hess_inv, dtype=float)

    # 8) Cholesky（上三角）
    Cu = linalg.cholesky(Sigma, lower=False)
    log_det_Cu = np.sum(np.log(np.diag(Cu)))

    data_slice_np = preds[:n_preds, :dim]
    target_np = target[:n_preds].astype(float)

    # 9) f(x)：x∈[0,1]^dim -> 对数
    def f(x):
        # 防极端 0/1
        x = np.clip(np.asarray(x, dtype=float), 1e-12, 1 - 1e-12)
        z = norm.ppf(x)                           # 标准正态
        transformed_beta = mu + Cu @ z            # 注意：这里沿用你原脚本用上三角 Cu 右乘 z 的写法
        # 对数似然（保持原式）
        log_lik_each = -np.log(1. + np.exp(-target_np * (data_slice_np @ transformed_beta)))
        prior_val = -(0.5 / (scale_prior**2)) * np.sum(transformed_beta**2)
        log_phi_z = np.sum(np.log(norm.pdf(z)))
        # 返回 log：|det C| + sum(loglik) + prior - log phi(z)
        return float(log_det_Cu + np.sum(log_lik_each) + prior_val - log_phi_z)

    return f, dim
