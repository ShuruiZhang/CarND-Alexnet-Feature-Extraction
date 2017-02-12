import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
# Load pickled data
import pickle

# Fill this in based on where you saved the training and testing data

training_file ='train.p'
testing_file ='test.p'

with open(training_file, mode='rb') as f:
    print("importing training file")
    train = pickle.load(f)
    print("file loaded")
with open(testing_file, mode='rb') as f:
    print("importing test file")
    test = pickle.load(f)
    print("file loaded")
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']
print("data loaded")
# TODO: Split data into training and validation sets.
from sklearn.model_selection import train_test_split
X_nn_train,X_validation,y_nn_train,y_validation= train_test_split(X_train,y_train,test_size=0.1,random_state=0)
# TODO: Define placeholders and resize operation.
features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))
# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

nb_classes = 43
# TODO: Add the final layer for traffic sign classification.
fc7 = AlexNet(resized, feature_extract=True)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
#define weights
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
#define bias
fc8b = tf.Variable(tf.zeros(nb_classes))
#get logits
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
#add activation
probs = tf.nn.softmax(logits)
# TODO: Define loss, training, accuracy operations.
cross_entropy= tf.nn.softmax_cross_entropy_with_logits(logits,one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)#get loss
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation=optimizer.minimize(loss_operation,var_list=[fc8W, fc8b]) 

init_op = tf.initialize_all_variables()

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

EPOCH = 10
batch_size = 128
# TODO: Train and evaluate the feature extraction model.
def eval_on_data(X, y, sess):
    total_acc = 0
    total_loss = 0
    for offset in range(0, X.shape[0], batch_size):
        end = offset + batch_size
        X_batch = X[offset:end]
        y_batch = y[offset:end]

        loss, acc = sess.run([loss_op, accuracy_op], feed_dict={features: X_batch, labels: y_batch})
        total_loss += (loss * X_batch.shape[0])
        total_acc += (acc * X_batch.shape[0])

    return total_loss/X.shape[0], total_acc/X.shape[0]

###training pipline below


with tf.Session() as sess:
    sess.run(init_op)

    for i in range(epochs):
        # training
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], batch_size):
            end = offset + batch_size
            sess.run(train_op, feed_dict={features: X_train[offset:end], labels: y_train[offset:end]})

        val_loss, val_acc = eval_on_data(X_val, y_val, sess)
        print("Epoch", i+1)
        print("Time: %.3f seconds" % (time.time() - t0))
        print("Validation Loss =", val_loss)
        print("Validation Accuracy =", val_acc)
        print("")