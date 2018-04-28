import csv
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
from pdb import set_trace as st

grid = True
test_file = "data/agr_en_fb_test.csv"

f = open("data/agr_en_train.csv")
stream = csv.DictReader(f, fieldnames=('title', 'body','topic'))
X_train, y_train = zip(*[(x['body'], x['topic']) for x in stream])
f.close()

f = open("data/agr_en_dev.csv")
stream = csv.DictReader(f, fieldnames=('title', 'body','topic'))
X_dev, y_dev = zip(*[(x['body'], x['topic']) for x in stream])
f.close()

f = open(test_file)
stream = csv.DictReader(f, fieldnames=('title', 'body'))
test_titles, X_test = zip(*[(x['title'], x['body'])  for x in stream])
f.close()

text_clfs = [
        Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', ngram_range=(1, 5), lowercase=True) ),
		             ('gauss', RBFSampler(random_state=1)),
                     ('clf', PassiveAggressiveClassifier() )
                ]),
        Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', ngram_range=(1, 5), lowercase=True) ),
                   
                     ('clf', PassiveAggressiveClassifier() )
                ]),
            ]

parameters = [{
#parameters =  {
            #'tfidf__ngram_range': [(1, 1), (1, 2)],#, (1,5), (1,4), (2, 4), (2, 5)],
                'tfidf__sublinear_tf': (True, False),
                'tfidf__stop_words': (None, 'english'),
                #'tfidf__lowercase': (True, False),
                #'tfidf__analyzer': ('word', 'char'),
                'tfidf__binary': (True, False),
	            'gauss__gamma': (2.0, 1.0, 0.1, 0.001, 0.0001, 0.00001),
                'clf__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0)

            },
           {#'tfidf__ngram_range': [(1, 1), (1, 2), (1,5), (1,4), (2, 4), (2, 5)],
                'tfidf__sublinear_tf': (True, False),
                'tfidf__stop_words': (None, 'english'),
           #     'tfidf__lowercase': (True, False),
           #     'tfidf__analyzer': ('word', 'char'),
                'tfidf__binary': (True, False),
                'clf__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0)
            }
    ]

if not grid:
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_dev)
else:
    performances=[]
    for text_clf, param in zip(text_clfs, parameters):
        gs_clf = GridSearchCV(text_clf, param, n_jobs=-1, scoring='f1_macro')
        gs_clf = gs_clf.fit(X_train, y_train)
        predicted = gs_clf.predict(X_dev)
        performances.append((gs_clf, predicted, f1_score(y_dev, predicted, average='macro')))

clf = sorted(performances, key=lambda x: x[2], reverse=True)[0]
gs_clf = clf[0]
predicted = clf[1]
pd.DataFrame(list(gs_clf.cv_results_.items())).to_csv("stats_PA_fb_en.csv")
# imprimir evaluacion de predicciones con el conjunto dev
print("F1_macro: %f\n" % clf[2])
print(classification_report(y_dev, predicted))

if grid:
    predicted = gs_clf.predict(X_test)
else:
    predicted = text_clf.predict(X_dev)
# imprimir prediccinoes con el conjunto test (competencia)
with open("results_PA_fb_en.csv", 'w') as f:
    f.write("%s\n" % gs_clf.best_estimator_)
    for id, l in zip(test_titles, predicted):
        f.write("%s,%s\n" % (id, l))
#metodo_para_imprimir_predicciones(predicted)
