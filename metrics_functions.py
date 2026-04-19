import numpy as np


#-----------------------------------------------------------#


def sensitivity(label, pred):

    """
    Sensitivity is also seen as true positive rate (TPR) and can be defined as the proportion of pixels correctly classified as 
    blood vessel pixels [45]. The range of sensitivity has a scale from 0 to 1.
    """

    pred = (pred > 127)
    label = (label > 127)

    TP = np.sum(pred & label)
    FN = np.sum(~pred & label)

    return TP / (TP + FN + 1e-7) # Aggiungo un 1e-7 per evitare divisioni per 0


#-----------------------------------------------------------#


def specificity(label, pred):

    """
    Ability to predict a true negative of each category available
    """

    pred = (pred > 127)
    label = (label > 127)

    TN = np.sum( ~pred & ~label )
    FP = np.sum( pred & ~label )
    return TN / (TN + FP + 1e-7)


#-----------------------------------------------------------#


def accuracy(label, ):

    pred = (pred > 127)
    label = (label > 127)

    TP = np.sum(pred & label)
    FN = np.sum(~pred & label)
    TN = np.sum( ~pred & ~label )
    FP = np.sum( pred & ~label )

    return (TP+TN) / (TP+TN+FP+FN+1e-7)


#-----------------------------------------------------------#


def false_positive_rate(label, pred):

    pred = (pred > 127)
    label = (label > 127)

    TN = np.sum( ~pred & ~label )
    FP = np.sum( pred & ~label )

    return FP / (TN+FP+1e-7)


#-----------------------------------------------------------#


def mcc(label, pred):

    """
    a robust evaluation metric used in vessel segmentation, particularly for retinal images, to handle severe class imbalance 
    where vessel pixels (foreground) are far less frequent than background pixels. . It acts as a correlation coefficient between 
    the observed (ground-truth) and predicted binary classifications, returning a value between -1 and +1, where +1 is perfect 
    prediction, 0 is no better than random, and -1 indicates total disagreement.
    """

    pred = (pred > 127).astype(np.float64)
    label = (label > 127).astype(np.float64)

    TP = np.sum(pred & label)
    FN = np.sum(~pred & label)
    TN = np.sum( ~pred & ~label)
    FP = np.sum( pred & ~label)

    return ( (TN*TP) - (FN*FP) ) / np.sqrt( (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN) )


#-----------------------------------------------------------#


def get_metrics(label,pred):
    

    pred = (pred > 127).astype(np.float64)
    label = (label > 127).astype(np.float64) 

    tp = np.sum(pred * label)
    tn = np.sum((1 - pred) * (1 - label))
    fp = np.sum(pred * (1 - label))
    fn = np.sum((1 - pred) * label)

    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    fpr = fp / (tn + fp + 1e-7)
    dice = (2*tp) / (2*tp + fp + fn + 1e-7) # 2 x Area of overlap / Total Area
    IoU = tp / (tp + fp + fn + 1e-7)        # area of overlap / area of union
    
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    if den == 0:
        mcc = 0.0
    else:
        mcc = num / den

    return sens, spec, acc, fpr, mcc, dice, IoU


#-----------------------------------------------------------#








    