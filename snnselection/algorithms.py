import pickle
import sklearn

from sklearn.utils import all_estimators

algorithms = {
    'cluster' : all_estimators(type_filter='cluster'),
    'regressor' : all_estimators(type_filter='regressor'),
    'classifier' : all_estimators(type_filter='classifier'),
}
