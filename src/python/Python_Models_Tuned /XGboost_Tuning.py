from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline

from data_prep import load_and_preprocess_data
from ROC_gen import ROC_Generator
from conf_matrix import Matrix_Display
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()


pipe = Pipeline([
        ('smotetomek', SMOTETomek(random_state=42)),
        ('clf', XGBClassifier(random_state=42))
        ])

search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001, 1.0, prior='log-uniform'),
    'clf__subsample': Real(0.5, 1.0),
    'clf__colsample_bytree': Real(0.5, 1.0),
    'clf__colsample_bylevel': Real(0.5, 1.0),
    'clf__colsample_bynode' : Real(0.5, 1.0),
    'clf__reg_alpha': Real(0.0, 10.0),
    'clf__reg_lambda': Real(0.0, 10.0),
    'clf__gamma': Real(0.0, 10.0),
    }

opt = BayesSearchCV(pipe, search_space, cv=15, n_iter=20, scoring='roc_auc', random_state=42)


