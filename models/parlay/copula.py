import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import brentq
def bvn(a,b,r): return multivariate_normal(mean=[0,0],cov=[[1,r],[r,1]]).cdf([a,b])
def tetra(pa,pb,pab):
    a,b=norm.ppf(pa),norm.ppf(pb)
    f=lambda r: bvn(a,b,r)-pab
    lo,hi=-0.99,0.99
    if f(lo)>0: return lo
    if f(hi)<0: return hi
    return brentq(f,lo,hi,xtol=1e-4)
def joint_gauss(p,R):
    """P(all legs hit) for legs with hit probs p and latent correlation matrix R (Gaussian copula)."""
    p=np.asarray(p); z=norm.ppf(p); R=np.asarray(R,float)
    w,V=np.linalg.eigh(R)
    if w.min()<1e-6: R=V@np.diag(np.clip(w,1e-6,None))@V.T; d=np.sqrt(np.diag(R)); R=R/np.outer(d,d)
    if len(p)==1: return float(p[0])
    return float(multivariate_normal(mean=np.zeros(len(p)),cov=R,allow_singular=True).cdf(z))
