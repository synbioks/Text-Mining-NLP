from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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
