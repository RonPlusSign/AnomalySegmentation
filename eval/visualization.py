import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from ood_metrics import aupr, auroc, fpr_at_95_tpr
import cv2

def save_colored_score_image(image_path, anomaly_score, save_path, file_name):
    """
    Save the image with the anomaly score colored in a new image.
    
    image_path: path to the input image
    anomaly_score: anomaly score for each pixel
    save_path: path to save the colored image
    """
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize the anomaly score
    anomaly_score = (anomaly_score - np.min(anomaly_score)) / (np.max(anomaly_score) - np.min(anomaly_score))
    
    # Apply the colormap
    anomaly_score = cv2.applyColorMap((anomaly_score * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Combine the original image and the colored anomaly score
    combined = cv2.addWeighted(image, 0.5, anomaly_score, 0.5, 0)
    
    # Save the image
    cv2.imwrite(f"{save_path}/{file_name}_colored_score_image.png", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def plot_roc(preds, labels, title="Receiver operating characteristic", save_path=None, file_name=None):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}/{file_name}_roc_curve.png")
    else:
        plt.show()


def plot_pr(preds, labels, title="Precision recall curve", save_path=None, file_name=None):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}/{file_name}_pr_curve.png")
    else:
        plt.show()


def plot_barcode(preds, labels, title="Barcode plot", save_path=None, file_name=None):
    """Plot a visualization showing inliers and outliers sorted by their prediction of novelty."""
    # the bar
    x = sorted([a for a in zip(preds, labels)], key=lambda x: x[0])
    x = np.array([[49, 163, 84] if a[1] == 1 else [173, 221, 142] for a in x])
    # x = np.array([a[1] for a in x]) # for bw image

    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary_r, interpolation='nearest')

    plt.title(title)
    fig = plt.figure()

    # a horizontal barcode
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x.reshape((1, -1, 3)), **barprops)
    plt.tight_layout()
    
    
    if save_path is not None:
        plt.savefig(f"{save_path}/{file_name}_barcode_plot.png")
    else:
        plt.show()