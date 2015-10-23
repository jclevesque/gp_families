
import scipy.stats as ss
import numpy as np


def expected_improvement(mu, s2, vals, dmu=None, ds2=None, grad=False):
    fmax = np.max(vals)

    s = np.sqrt(s2)
    d = mu - fmax

    not_zero_s = (s != 0)

    z = d[not_zero_s] / s[not_zero_s]

    pdfz = ss.norm.pdf(z)
    cdfz = ss.norm.cdf(z)
    ei = d.copy()
    ei[not_zero_s] = d[not_zero_s] * cdfz + s[not_zero_s] * pdfz
    #Other EI values remain at d because there isn't any variance

    if grad:
        # get the derivative of ei. The mu/s2/etc. components are vectors
        # collecting n scalar points, whereas dmu and ds2 are (n,d)-arrays.
        # The indexing tricks just interpret the "scalar" quantities as
        # (n,1)-arrays so that we can use numpy's broadcasting rules.
        dei = 0.5 * ds2 / s2[:, None]
        dei *= (ei - s * z * cdfz)[:, None]
        dei += cdfz[:, None] * dmu
        return ei, dei
    else:
        return ei
