import numpy as np

class LogReg:
    def __init__(self, X, y, n, m, batch_size, reg, reg_coef):
        self.X = X
        self.y = y
        self.n = n
        self.m = m
        self.d = X.shape[1]
        self.batch_size = batch_size
        self.reg = reg
        self.lmb = reg_coef

    def sigmoid(t):
        return 1.0 / (1 + np.exp(-t))

    def local_gradient(self, x, i, rng=None):
        m = self.m
        left = i*m
        right = (i+1)*m

        if self.batch_size is None or rng is None:
            sample_idx = np.array(range(left, right))
        else:
            sample_idx = rng.choice(np.array(range(left, right)), size=self.batch_size, replace=False)

        z = self.y[sample_idx] * np.dot(self.X[sample_idx], x)
        tmp0 = np.minimum(z, 0)
        tmp1 = np.exp(-z) / ((1+ np.exp(-z)))
        tmp2 = - tmp1 * self.y[sample_idx]
        gradient = np.dot(self.X[sample_idx].T, tmp2)

        if self.batch_size is None:
            gradient /= m
        else:
            gradient /= self.batch_size

        gradient += self.lmb*self.reg(x)

        return gradient

    def full_gradient(self, x):
        z = self.y * np.dot(self.X, x)
        tmp0 = np.minimum(z, 0)
        tmp1 = np.exp(-z) / ((1+ np.exp(-z)))
        tmp2 = - tmp1 * self.y
        gradient = np.dot(self.X.T, tmp2)

        gradient /= (self.n * self.m)
        gradient += self.lmb*self.reg(x)

        return gradient