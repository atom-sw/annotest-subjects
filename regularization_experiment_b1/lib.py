from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt

'''
Parameters
    ------------
    soft : bool, optional
        Whether to plot soft regions. Default is False. Only applicable to binary classifiers.
'''
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, soft=False):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    if soft:
        Z = classifier.predict_proba(np.array([xx1.ravel(), xx2.ravel()]).T)[:, 0]
        Z = Z.reshape(xx1.shape)
        contour = plt.contourf(xx1, xx2, Z, alpha=0.4, camp=colors[0])
        plt.colorbar(contour)
    else:
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')

def gen_data(num_data, sigma):
    x = 2 * np.pi * (np.random.rand(num_data) - 0.5)
    y = np.sin(x) + np.random.normal(0, sigma, num_data)
    return (x, y)

######### preprocessor ############
import re

def preprocessor(text):
    text = re.sub('[0-9,]+', 'NUM', text.lower())
    text.replace('&nbsp;', ' ')

    r = '(?::|;|=|X)(?:-)?(?:\)|\(|D|P)'
    emoticons = re.findall(r, text)
    text = re.sub(r, '', text)

    text = re.sub('[\W]+', ' ', text.lower()) + ' ' + ' '.join(emoticons).replace('-','')
    return text

########### stop-words ############
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
eng_stop = stopwords.words('english')

def tokenizer_stem_nostop(text):
    porter = PorterStemmer()
    return [w for w in text.split(' ') if w not in eng_stop]

########### plot history ############
def plot_history(his):
    train_loss = his.history['loss']
    val_loss = his.history['val_loss']

    # visualize training history
    plt.plot(range(1, len(train_loss)+1), train_loss, color='blue', label='Train loss')
    plt.plot(range(1, len(val_loss)+1), val_loss, color='red', label='Val loss')
    plt.legend(loc="upper right")
    plt.xlabel('#Epoch')
    plt.ylabel('Loss')
    plt.savefig('./output/fig-nn-val.png', dpi=300)
    plt.show()
