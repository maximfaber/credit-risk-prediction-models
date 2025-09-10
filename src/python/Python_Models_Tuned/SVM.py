
from sklearn.svm import SVC

from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

from skopt import BayesSearchCV
from skopt.space import Real, Categorical

from data_prep import load_and_preprocess_data
from ROC_gen import ROC_Generator
from conf_matrix import Matrix_Display

X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()


pipe = Pipeline([
        ('smotetomek', SMOTETomek(random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
])

search_space = {
    'svm__C': Real(0.01, 10, prior='log-uniform'),            # Regularization parameter
    'svm__gamma': Real(0.0001, 1.0, prior='log-uniform'),     # Kernel coefficient
    'svm__kernel': Categorical(['rbf'])                       # Kernel type
}
opt = BayesSearchCV(
    estimator=pipe,
    search_spaces=search_space,
    n_iter=20,
    cv=10,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1,
    random_state=42
)
opt.fit(X_train, y_train)