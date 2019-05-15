

def test_works_on_MNIST():

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.layers_custom = self.create_layers()

        def create_layers(self):
            layers = [Dense(128, activation='relu'), Dense(10, activation='softmax')]

            return layers

        def call(self, x):
            x = tf.keras.layers.Flatten(input_shape=(28, 28))(x)
            for layer in self.layers_custom:
                x = layer(x)
            return x

    model = MyModel()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Flatten(input_shape=(28, 28)),
    #   tf.keras.layers.Dense(128, activation='relu'),
    #   tf.keras.layers.Dropout(0.2),
    #   tf.keras.layers.Dense(10, activation='softmax')
    # ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test, y_test)

    results = model.evaluate(x_test, y_test)
    assert results[1] > 0.9