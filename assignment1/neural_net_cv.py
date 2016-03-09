# A bit of setup

import numpy as np
import matplotlib.pyplot as plt
import  copy

from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.vis_utils import visualize_grid

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10

best_net = None # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
results = {}
best_val = -1
learning_rates = [1e-3]
regularization_strengths = [0.05,0.1,0.5]
hidden_size = [50,100]

for lr in learning_rates:
    for reg in regularization_strengths:
        for hnum in hidden_size:
            net = TwoLayerNet(input_size, hnum, num_classes)
            stats = net.train(X_train, y_train, X_val, y_val,
                num_iters=1000, batch_size=200,
                learning_rate=lr, learning_rate_decay=0.95,
                reg=reg, verbose=True)
            y_train_pred = net.predict(X_train)
            y_val_pred = net.predict(X_val)
            tmp_train_accuracy=np.mean(y_train == y_train_pred)
            tmp_val_accuracy=np.mean(y_val == y_val_pred)
            results[(lr,reg,hnum)]=[tmp_train_accuracy,tmp_val_accuracy]
            if tmp_val_accuracy>best_val:
                best_val=tmp_val_accuracy
                best_net=copy.deepcopy(net)
#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################

# Print out results.
for lr, reg,hnum in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg,hnum)]
    print 'lr %e reg %e hiddennum %d train accuracy: %f val accuracy: %f' % (
                lr, reg,hnum, train_accuracy, val_accuracy)

print 'best validation accuracy achieved during cross-validation: %f' % best_val


# visualize the weights of the best network
def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
  plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
  plt.gca().axis('off')
  plt.show()

show_net_weights(best_net)


test_acc = (best_net.predict(X_test) == y_test).mean()
print 'Test accuracy: ', test_acc



