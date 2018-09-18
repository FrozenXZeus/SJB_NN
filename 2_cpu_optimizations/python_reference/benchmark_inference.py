from sjb_nn import SJB_NN
from chainer import serializers
import chainer
import chainer.functions as F
import numpy as np
from chainer.dataset import concat_examples
import time

model = SJB_NN()

time_start = time.perf_counter()

train, test = chainer.datasets.get_fashion_mnist()
test_data, test_labels = concat_examples(test)

time_dataset_load = time.perf_counter()

serializers.load_npz('sjb_nn.model', model)


l1_w = model.l1.W.array
l1_b = model.l1.b.array

l2_w = model.l3.W.array
l2_b = model.l3.b.array

time_model_load = time.perf_counter()

layer_1_res = np.matmul(test_data, l1_w.T) + l1_b;

layer_1_output = F.relu(layer_1_res)

calculations = np.matmul(layer_1_output.data, l2_w.T) + l2_b

time_forward_prop = time.perf_counter()

prediction_vals = F.softmax(calculations, axis=1)
prediction_vals.backward()
final_predictions = np.argmax(prediction_vals.data, axis=1)

time_predictions = time.perf_counter()

#print("First 10 predictions:\n{}".format("\t".join(map(str, final_predictions[:10]))))

print("Total Time taken: {:.3f} ms".format((time_predictions - time_start)*1000))
print("Dataset load: {:.3f} ms".format((time_dataset_load-time_start)*1000))
print("Weight load: {:.3f} ms".format((time_model_load-time_dataset_load)*1000))
print("Forward Prop: {:.3f} ms".format((time_forward_prop - time_model_load)*1000))
print("Prediction: {:.3f} ms".format((time_predictions-time_forward_prop)*1000))