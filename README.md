An LSTM (RNN) implementation by using Keras and Tensoflow. The model can be adopted to any project based on need.

Problem definition:

A model is implemented where 5 consecutive (integer) numbers are given and the 6th one is estimated by the model.

Example: 
Input: [[1], [2], [3], [4], [5]]
Expected Output: 6

As seen in the above description and example, the length of sequence in each (train) example is 5. It can be read as the total timestep that we run through in each step is 5 as well.

The entire dataset consists of 200 examples starting with (Input: [[1], [2], [3], [4], [5]], Output: 6) and ending with (Input: [[200], [201], [202], [203], [204]], Output: 205).

The data is pre-normalized before usage and each number in the dataset is just divided by 205.

The dataset is also divided into train, dev, and test sets. The distribution is as following:
- Train: 140 examples
- Dev: 40 examples
- Test: 20 examples


The data is saved in corresponding csv files and there is a separate csv file for each set. Reading the data is done by using Tensorflow's initializable Iterator from the Dataset API. A data generator is used in all training, validation and test phases and training is done with Keras' fit_generator(...) function.

Lastly, a number of Keras callback functions are added to the training. Any of them can manually be deactivated if no value is seen. Implemented callbacks are:
- TerminateOnNaN    - Terminate training if NaN loss is encountered
- ModelCheckpoint   - Save the model as configured in the code
- EarlyStopping     - Stop training if no improvement is seen based on settings in the function call
- ReduceLROnPlateau - Reduce learning rate (alpha) if the specified monitored metric does not improved as
                      configured in the function call
