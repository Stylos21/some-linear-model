//npm packages
const tf = require('@tensorflow/tfjs');
const rline = require('readline');
//init readline interface

const rl = rline.createInterface({
  input: process.stdin,
  output: process.stdout
});


//make an ML model
const model = tf.sequential();
  //add layer
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
// Compile model for training, and then specify loss and optimizer
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    // random data
    const xs = tf.tensor1d([5.303, 8.320, 1.574, 0.284, 6.076, 3.234, 1.263, 3.926, 1.567, 3.383, 6.747, 4.376, 9.034, 4.193, 2.331, 1.001, 0.106, 5.506, 0.167, 8.658, 9.271, 9.830, 7.812, 1.841, 9.245]);
    const ys = tf.tensor1d([3.428, 8.098, 5.141, 8.737, 5.251, 3.072, 5.653, 2.056, 5.210, 8.369, 4.257, 1.993, 3.943, 0.681, 2.468, 5.606, 8.187, 5.405, 7.451, 0.168, 5.028, 3.123, 2.603, 2.053, 6.187]);

    rl.question('Welcome to this random linear regression model! Please input an answer. ', (answer) => {

      model.fit(xs, ys).then(() => {
        model.predict(tf.tensor2d([answer], [1,1])).print();
      });
  rl.close();
});
