import time

import ArrayGenerator


def learn(validation=False):
    array_generator = ArrayGenerator.ArrayGenerator()
    start = time.time()
    X, y = array_generator.generate_inputs_labels('data/train/modified', 50)
    end = time.time()
    print("input generation time: {}".format(end-start))
    print("X: {}".format(X.shape))
    print("y: {}".format(y.shape))

    # to do: create training and validation sets
    # learn
    if validation:
        pass
    else:
        pass


    # predict
    # save

learn()