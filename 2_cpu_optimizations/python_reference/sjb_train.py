import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import optimizers
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu

from sjb_nn import SJB_NN

train, test = chainer.datasets.get_fashion_mnist()

train_iter = iterators.SerialIterator(train, 60000)
test_iter = iterators.SerialIterator(test, 10000, False, False)

gpu_id = 0
model = SJB_NN()

max_epoch = 250

classifier = L.Classifier(model)

# selection of your optimizing method
optimizer = optimizers.MomentumSGD()

# Give the optimizer a reference to the model
optimizer.setup(classifier)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0003))

# Get an updater that uses the Iterator and Optimizer
updater = training.updaters.StandardUpdater(train_iter, optimizer)

trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

trainer.extend(extensions.LogReport())
trainer.extend(extensions.Evaluator(test_iter, classifier, device=gpu_id))
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()

serializers.save_npz("sjb_nn.model", model)