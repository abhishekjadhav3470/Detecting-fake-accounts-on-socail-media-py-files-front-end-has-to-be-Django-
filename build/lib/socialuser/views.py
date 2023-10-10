from django.shortcuts import render,redirect

# Create your views here.
from socialuser.models import user_reg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import gender_guesser.detector as gender

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, BatchNormalization, Activation, Dropout
from keras.callbacks import EarlyStopping, Callback
import matplotlib.pyplot as plt
from IPython.display import clear_output
def socialuser_index(request):

    return render(request,'socialuser/socialuser_index.html')


def socialuser_login(request):
    if request.method == "POST":
        uname = request.POST.get('uname')
        pswd = request.POST.get('password')
        try:
            check = user_reg.objects.get(uname=uname,password=pswd)
            print(check)
            request.session['user_id'] = check.id
            request.session['user_name'] = check.uname
            return redirect('socialuser_home')
        except:
            pass
        return redirect('socialuser_login')
    return render(request,'socialuser/socialuser_login.html')


def socialuser_register(request):
    if request.method == "POST":
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        mobile = request.POST.get('mobile')
        gender = request.POST.get('gender')
        place = request.POST.get('place')
        uname = request.POST.get('uname')
        password = request.POST.get('password')
        user_reg.objects.create(full_name=full_name,email=email, mobile=mobile, gender=gender, place=place,
                                    uname=uname, password=password)
        return redirect('socialuser_login')
    return render(request,'socialuser/socialuser_register.html')

def socialuser_home(request):
    if request.method == "POST" and request.FILES['dataset']:
        file = request.FILES['dataset']
        filename = file.name
        request.session['filename'] = filename
        print(filename)

        return redirect('csvdataview')
    return render(request,'socialuser/socialuser_home.html')


def csvdataview(request):
    fname = request.session['filename']
    print(fname)
    a = pd.read_csv(fname)
    data_html = a.to_html()
    context = {'loaded_data': data_html}
    return render(request,'socialuser/csvdataview.html',context)



def svm_algorithm(request):
    best1=''
    def read_datasets():
        """ Reads users profile from csv files """
        genuine_users = pd.read_csv("users.csv")
        fake_users = pd.read_csv("fusers.csv")
        # print genuine_users.columns
        # print genuine_users.describe()
        # print fake_users.describe()
        x = pd.concat([genuine_users, fake_users])
        y = len(fake_users) * [0] + len(genuine_users) * [1]
        return x, y

    def extract_features(x):
        lang_list = list(enumerate(np.unique(x['lang'])))
        lang_dict = {name: i for i, name in lang_list}
        x.loc[:, 'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)

        feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count',
                                  'listed_count', 'lang_code']
        x = x.loc[:, feature_columns_to_use]
        return x

    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        target_names = ['Fake', 'Genuine']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_roc_curve(y_test, y_pred):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        print("False Positive rate: ", false_positive_rate)
        print("True Positive rate: ", true_positive_rate)

        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.title('SVM Algorithm(Support Vector Machine')
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def train(X_train, y_train, X_test):
        """ Trains and predicts dataset with a SVM classifier """
        # Scaling features
        X_train = preprocessing.scale(X_train)
        X_test = preprocessing.scale(X_test)

        Cs = 10.0 ** np.arange(-2, 3, .5)
        gammas = 10.0 ** np.arange(-2, 3, .5)
        param = [{'gamma': gammas, 'C': Cs}]
        cvk = StratifiedKFold(n_splits=5)
        classifier = SVC()
        clf = GridSearchCV(classifier, param_grid=param, cv=cvk)
        clf.fit(X_train, y_train)
        best1 = clf.best_estimator_
        print("The best classifier is: ", clf.best_estimator_)
        clf.best_estimator_.fit(X_train, y_train)
        # Estimate score
        scores = cross_val_score(clf.best_estimator_, X_train, y_train, cv=5)
        print(scores)
        print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
        title = 'Learning Curves (SVM, rbf kernel, $\gamma=%.6f$)' % clf.best_estimator_.gamma
        plot_learning_curve(clf.best_estimator_, title, X_train, y_train, cv=5)
        plt.show()
        # Predict class
        y_pred = clf.best_estimator_.predict(X_test)
        return y_test, y_pred

    x, y = read_datasets()

    # In[77]:

    print("extracting featues.....\n")
    x = extract_features(x)
    print(x.columns)
    print(x.describe())

    # In[78]:

    print("spliting datasets in train and test dataset...\n")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)

    # In[79]:

    print("training datasets.......\n")
    y_test, y_pred = train(X_train, y_train, X_test)

    # In[80]:

    print('Classification Accuracy on Test dataset: ', accuracy_score(y_test, y_pred))

    # In[82]:

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)
    plot_confusion_matrix(cm)

    # In[83]:

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    # In[84]:
    aa = classification_report(y_test, y_pred, target_names=['Fake', 'Genuine'])
    context = {'loaded_data': aa}
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))

    # In[85]:

    plot_roc_curve(y_test, y_pred)
    return render(request,'socialuser/svm_algorithm.html')

def random_forest(request):
    def read_datasets():
        """ Reads users profile from csv files """
        genuine_users = pd.read_csv("users.csv")
        fake_users = pd.read_csv("fusers.csv")
        # print genuine_users.columns
        # print genuine_users.describe()
        # print fake_users.describe()
        x = pd.concat([genuine_users, fake_users])
        y = len(fake_users) * [0] + len(genuine_users) * [1]
        return x, y

        ####### function for predicting sex using name of person

        # In[56]:

        ####### function for feature engineering

        # In[57]:

    def extract_features(x):
        lang_list = list(enumerate(np.unique(x['lang'])))
        lang_dict = {name: i for i, name in lang_list}
        x.loc[:, 'lang_code'] = x['lang'].map(lambda x: lang_dict[x]).astype(int)

        feature_columns_to_use = ['statuses_count', 'followers_count', 'friends_count', 'favourites_count',
                                  'listed_count',
                                  'lang_code']
        x = x.loc[:, feature_columns_to_use]
        return x

        ####### function for ploting learning curve

        # In[60]:

    def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                            n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

        ####### function for plotting confusion matrix

        # In[61]:

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        target_names = ['Fake', 'Genuine']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        ####### function for plotting ROC curve

        # In[62]:

    def plot_roc_curve(y_test, y_pred):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

        print("False Positive rate: ", false_positive_rate)
        print("True Positive rate: ", true_positive_rate)

        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.title('Random Forest Algorithm')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.2])
        plt.ylim([-0.1, 1.2])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        ####### Function for training data using Random Forest

        # In[63]:

    def train(X_train, y_train, X_test):
        """ Trains and predicts dataset with a Random Forest classifier """

        clf = RandomForestClassifier(n_estimators=40, oob_score=True)
        clf.fit(X_train, y_train)
        print("The best classifier is: ", clf)
        # Estimate score
        scores = cross_val_score(clf, X_train, y_train, cv=5)
        print(scores)
        print('Estimated score: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))
        title = 'Learning Curves (Random Forest)'
        plot_learning_curve(clf, title, X_train, y_train, cv=5)
        plt.show()
        # Predict
        y_pred = clf.predict(X_test)
        return y_test, y_pred

        # In[64]:

    print("reading datasets.....\n")
    x, y = read_datasets()
    x.describe()

    # In[65]:

    print("extracting featues.....\n")
    x = extract_features(x)
    print(x.columns)
    print(x.describe())

    # In[66]:

    print("spliting datasets in train and test dataset...\n")
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=44)

    # In[67]:

    print("training datasets.......\n")
    y_test, y_pred = train(X_train, y_train, X_test)

    # In[68]:

    print('Classification Accuracy on Test dataset: ', accuracy_score(y_test, y_pred))

    # In[70]:

    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix, without normalization')
    print(cm)
    plot_confusion_matrix(cm)

    # In[71]:

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

    # In[72]:

    print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine']))

    # In[73]:

    plot_roc_curve(y_test, y_pred)
    return render(request,'socialuser/random_forest.html')


def neural_network(request):
    data = dict()
    data["fake"] = pd.read_csv("fusers.csv")
    data["legit"] = pd.read_csv("users.csv")

    data["legit"] = data["legit"].drop(
        ["id", "name", "screen_name", "created_at", "lang", "location", "default_profile", "default_profile_image",
         "geo_enabled", "profile_image_url", "profile_banner_url", "profile_use_background_image",
         "profile_background_image_url_https", "profile_text_color", "profile_image_url_https",
         "profile_sidebar_border_color", "profile_background_tile", "profile_sidebar_fill_color",
         "profile_background_image_url", "profile_background_color", "profile_link_color", "utc_offset", "protected",
         "verified", "dataset", "updated", "description"], axis=1)
    data["fake"] = data["fake"].drop(
        ["id", "name", "screen_name", "created_at", "lang", "location", "default_profile", "default_profile_image",
         "geo_enabled", "profile_image_url", "profile_banner_url", "profile_use_background_image",
         "profile_background_image_url_https", "profile_text_color", "profile_image_url_https",
         "profile_sidebar_border_color", "profile_background_tile", "profile_sidebar_fill_color",
         "profile_background_image_url", "profile_background_color", "profile_link_color", "utc_offset", "protected",
         "verified", "dataset", "updated", "description"], axis=1)

    print("Final Available Columns")
    print(data["legit"].columns)

    data["legit"] = data["legit"].values
    data["fake"] = data["fake"].values

    for i in range(len(data["legit"])):
        if type(data["legit"][i][5]) == str:
            data["legit"][i][5] = 1

        if type(data["legit"][i][6]) == str:
            data["legit"][i][6] = 1

    for i in range(len(data["fake"])):
        if type(data["fake"][i][5]) == str:
            data["fake"][i][5] = 1

        if type(data["fake"][i][6]) == str:
            data["fake"][i][6] = 1

    data["legit"] = data["legit"].astype(np.float64)
    data["fake"] = data["fake"].astype(np.float64)

    where_nans = np.isnan(data["legit"])
    data["legit"][where_nans] = 0

    where_nans = np.isnan(data["fake"])
    data["fake"][where_nans] = 0

    X = np.zeros((len(data["fake"]) + len(data["legit"]), 7))
    Y = np.zeros(len(data["fake"]) + len(data["legit"]))

    for i in range(len(data["legit"])):
        X[i] = data["legit"][i] / max(data["legit"][i])
        Y[i] = -1

    for i in range(len(data["fake"])):
        bound = max(data["fake"][i])
        if bound == 0:
            bound = 1

        X[len(data["legit"]) + i] = data["fake"][i] / bound  # Normalizing Data [0 <--> 1]
        Y[len(data["legit"]) + i] = 1

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.24, random_state=42)

    early_stopping = EarlyStopping(monitor='val_loss', patience=2)

    class PlotLearning(Callback):
        def on_train_begin(self, logs={}):
            self.i = 0
            self.x = []
            self.losses = []
            self.val_losses = []
            self.acc = []
            self.val_acc = []
            self.fig = plt.figure()

            self.logs = []

        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(logs)
            self.x.append(self.i)
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))
            self.i += 1
            f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

            clear_output(wait=True)

            ax1.set_yscale('Log')
            ax1.plot(self.x, self.losses, label="loss")
            ax1.plot(self.x, self.val_losses, label="val_loss")
            ax1.legend()

            ax2.plot(self.x, self.acc, label="accuracy")
            ax2.plot(self.x, self.val_acc, label="validation accuracy")
            ax2.legend()

            plt.show();

    plot = PlotLearning()

    model = Sequential([
        BatchNormalization(),

        Dense(16, activation="relu", kernel_regularizer="l2"),
        BatchNormalization(),
        Dense(8, activation="relu", kernel_regularizer="l2"),
        BatchNormalization(),
        Dense(1, activation="tanh"),
    ])

    model.build((None, X.shape[1]))
    model.summary()
    model.compile(
        optimizer="adadelta",
        loss="binary_crossentropy",
        metrics=["acc"]
    )

    logits = model.predict(X_test).T[0]

    for i in range(len(logits)):
        logits[i] = -1 if logits[i] < 0 else 1

    def plot_confusion_matrix(cm, title='Neural Network', cmap=plt.cm.Reds):
        target_names = ['Fake', 'Genuine']
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    mat = confusion_matrix(y_test, logits)
    print(mat)

    plot_confusion_matrix(mat)

    plt.show()

    train_acc = model.evaluate(X_train, y_train)
    val_acc = model.evaluate(X_test, y_test)
    print("Train Accuracy:", train_acc)
    print("Validation Accuracy:", val_acc)
    return render(request,'socialuser/neural_network.html')