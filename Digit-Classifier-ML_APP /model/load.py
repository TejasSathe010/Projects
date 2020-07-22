import tensorflow as tf


def init():

	reconstructed_model = tf.keras.models.load_model("./model/my_mnist_h5_model.h5")
	print("Loaded Model from disk")

	# loss,accuracy = reconstructed_model.evaluate(X_test,y_test)
	# print('loss:', loss)
	# print('accuracy:', accuracy)

	return reconstructed_model