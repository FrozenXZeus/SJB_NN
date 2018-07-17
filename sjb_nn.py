from chainer import Chain
import chainer.functions as F
import chainer.links as L

class SJB_NN(Chain):
    def __init__(self, n_mid_units=100, n_out=10):
        super(SJB_NN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        h = F.relu(self.l1(x))
        return self.l2(h)
