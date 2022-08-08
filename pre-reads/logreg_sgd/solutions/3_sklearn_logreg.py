######################
# TODO

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1e20)
clf.fit(X, y)

show_decision_boundary(clf.predict_proba, data=(X, y))
clf.score(X, y)

# END TODO
#######################
