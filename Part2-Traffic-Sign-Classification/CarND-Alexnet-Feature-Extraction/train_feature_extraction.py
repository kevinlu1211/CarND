# import pickle
# import tensorflow as tf
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# from alexnet import AlexNet
# 
# # Load traffic signs data.
# training_file = "train.p"
# 
# with open(training_file, mode = 'rb') as f:
#     data = pickle.load(f)
# features = data['features']
# labels = data['labels']
# 
# # Split data into training and validation sets.
# train_feature, validation_feature, train_labels, validation_labels = train_test_split(features, labels, test_size = 0.2)
# 
# # Define placeholders and resize operation.
# original = tf.placeholder(tf.float32, (None, 32, 32, 3))
# resized = tf.image.resize_images(original, (227, 227))
# 
# # pass placeholder as first argument to `AlexNet`.
# fc7 = AlexNet(resized, feature_extract=True)
# 
# # use `tf.stop_gradient` to prevent the gradient from flowing backwards
# # past this point, keeping the weights before and up to `fc7` frozen.
# # This also makes training faster, less work to do!
# fc7 = tf.stop_gradient(fc7)
# 
# # Add the final layer for traffic sign classification.
# 
# # First find the size of the weight matrix for the last layer
# n_classes = 43
# fc7_width = fc7.get_shape().as_list()[-1]
# shape = (fc7_width, 43)  # use this shape for the weight matrix
# 
# # Define the weight and bias matrix
# weights_8 = tf.Variable(tf.truncated_normal(shape, stddev = 1e-2)) 
# bias_8 = tf.Variable(tf.zeros(n_classes))
# 
# # Add logits layer into the graph
# logits = tf.matmul(fc7, weights_8) + bias_8
# 
# # TODO: Define loss, training, accuracy operations.
# y = tf.placeholder(tf.int32, (None))
# y_one_hot = tf.one_hot(y, 43)
# 
# # Setup loss function and loss value
# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot)
# loss_operation = tf.reduce_mean(cross_entropy)
# 
# # Define training operations
# optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
# training_operation = optimizer.minimize(loss_operation, var_list = [weights_8, bias_8])
# 
# # Define cost metrics 
# correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
# accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 
# 
# # Train and evaluate the feature extraction model.
# BATCH_SIZE = 64
# EPOCHS = 10
# SAVE_FILE = os.getcwd()+"/alexnet_traffic_light"
# LOAD = False
# saver = tf.train.Saver()
# 
# # First write a helper function for validation set evaluation
# def evaluate(validation_feature, validation_labels):
# 	n_samples = len(validation_feature)
# 	total_accuracy = 0
# 	sess = tf.get_default_session()
# 	for offset in range(0, n_samples, BATCH_SIZE):
# 		batch_feature, batch_labels = validation_feature[offset:offset+BATCH_SIZE], \
# 										validation_labels[offset:offset+BATCH_SIZE]
# 		accuracy = sess.run(accuracy_operation, feed_dict = {original : batch_feature, \
# 													y: batch_labels})
# 		total_accuracy += (accuracy * len(batch_feature))
# 	return total_accuracy/n_samples
# 
# 
# # Now do the training operation
# with tf.Session() as sess:
# 	n_samples = len(train_feature)
# 	if LOAD:
# 		saver.restore(sess, SAVE_FILE)
# 		for i in range(EPOCHS):
# 			train_feature, train_labels = shuffle(train_feature, train_labels)
# 			for offset in range(0, n_samples, BATCH_SIZE):
# 				batch_feature, batch_labels = train_feature[offset:offset+BATCH_SIZE], \
# 				  train_labels[offset:offset+BATCH_SIZE]
# 				sess.run(training_operation, feed_dict = {original : batch_feature, \
# 														  y : batch_labels})
# 			validation_accuracy = evaluate(validation_feature, validation_labels)
# 			print("Validation accuracy for epoch {0}: {1}".format(i, validation_accuracy))	
# 		
# 		# Save the model
# 		saver.save(sess, SAVE_FILE)
# 	else:
# 		sess.run(tf.global_variables_initializer())	
# 		for i in range(EPOCHS):
# 			train_feature, train_labels = shuffle(train_feature, train_labels)
# 			for offset in range(0, n_samples, BATCH_SIZE):
# 				batch_feature, batch_labels = train_feature[offset:offset+BATCH_SIZE], \
# 				  train_labels[offset:offset+BATCH_SIZE]
# 				sess.run(training_operation, feed_dict = {original : batch_feature, \
# 														  y : batch_labels})
# 			validation_accuracy = evaluate(validation_feature, validation_labels)
# 			print("Validation accuracy for epoch {0}: {1}".format(i, validation_accuracy))	
# 		
# 		# Save the model
# 		saver.save(sess, SAVE_FILE)
# 	
import pickle
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from alexnet import AlexNet

nb_classes = 43
epochs = 10
batch_size = 128

with open('./train.p', 'rb') as f:
    data = pickle.load(f)

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.33, random_state=0)

features = tf.placeholder(tf.float32, (None, 32, 32, 3))
labels = tf.placeholder(tf.int64, None)
resized = tf.image.resize_images(features, (227, 227))

# Returns the second final layer of the AlexNet model,
# this allows us to redo the last layer for the traffic signs
# model.
fc7 = AlexNet(resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer()
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])
init_op = tf.initialize_all_variables()

preds = tf.arg_max(logits, 1)
accuracy_op = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


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