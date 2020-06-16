import numpy as np

from data_utils import Dataset
import data_gen as DG
from model import BiLSTMModel

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score



import argparse
parser = argparse.ArgumentParser(description="Hyperparameters")

parser.add_argument('--BATCH_SIZE', default=64)
parser.add_argument('--NUM_EPOCHS', default=4)
parser.add_argument('--EMBEDDING', default=300)
parser.add_argument('--MAX_SEQ_LEN', default=100)
parser.add_argument('--LSTM_HIDDEN_UNITS', default=100)
parser.add_argument('--LSTM_DENSE_DIM', default=200)

parser.add_argument('--DATA_DIR', default="data")

args = parser.parse_args()

def main():


    data = Dataset(args.DATA_DIR)
    sents, tags = data.get_all_data()

    MyModel = BiLSTMModel(args.MAX_SEQ_LEN, args.EMBEDDING, args.LSTM_HIDDEN_UNITS, args.LSTM_DENSE_DIM, data.get_nwords(), data.get_ntags())
    model = MyModel.define_model()

    # Datasets
    num_train_sents = len(data.train_sents)
    num_val_sents = len(data.val_sents)
    num_test_sents = len(data.test_sents)

    print("# train sents = {0} \n # of val sents = {1} \n # of test sents = {2}".format(
        num_train_sents, num_val_sents, num_test_sents), flush=True)

    partition = {"train": list(range(num_train_sents)), 
            "val": list(range(num_val_sents))}

    # Parameters
    params = {'dim': args.MAX_SEQ_LEN,
              'batch_size': args.BATCH_SIZE,
              'n_classes': data.get_ntags(),
              'shuffle': True,
              'word2idx': data.get_word2idx(),
              'tag2idx': data.get_tag2idx()}

    # Generators
    training_generator = DG.DataGenerator(partition['train'], data.train_sents, data.train_tags, **params)
    validation_generator = DG.DataGenerator(partition['val'], data.val_sents, data.val_tags, **params)


    # Train model on dataset
    history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        epochs=args.NUM_EPOCHS, 
                        verbose=1)

    # Parameters
    params_test = {'dim': args.MAX_SEQ_LEN,
              'batch_size': 1,
              'n_classes': data.get_ntags(),
              'shuffle': False,
              'word2idx': data.get_word2idx(),
              'tag2idx': data.get_tag2idx()}

    # Make predictions
    partition = {"test": list(range(num_test_sents))}
    testing_generator = DG.DataGenerator(partition['test'], data.test_sents, data.train_tags, **params_test)

    pred_test = model.predict_generator(generator=testing_generator,
                                        steps = num_test_sents)
    pred_test = np.argmax(pred_test, axis=-1)

    # print(pred_test.shape)

    def pad(x):
        x1 = [tgs + ([data.get_tag2idx()["PAD"]] * (args.MAX_SEQ_LEN - len(tgs))) for tgs in x]
        x2 = [tgs[:args.MAX_SEQ_LEN] for tgs in x1]
        return np.array(x2)

    test_tags_padded = pad(data.test_tags)
    # print(test_tags_padded.shape)

    def get_measures(yTrue, yPred):
        y1 = yTrue.reshape(1,-1).squeeze()
        y2 = yPred.reshape(1,-1).squeeze()

        P = precision_score(y1, y2, average=None)
        R = recall_score(y1, y2, average=None)
        F1 = f1_score(y1, y2, average=None)

        print("Precision=", flush=True)
        print(P, flush=True)
        print("Recall=", flush=True)
        print(R, flush=True)
        print("F1 score=", flush=True)
        print(F1, flush=True)


    print("Test...", flush=True)
    get_measures(test_tags_padded, pred_test)



if __name__ == "__main__":
    main()

