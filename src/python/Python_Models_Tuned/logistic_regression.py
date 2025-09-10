from sklearn.linear_model import LogisticRegression
from data_prep import load_and_preprocess_data
from sklearn.pipeline import Pipeline
from ROC_gen import ROC_Generator
from conf_matrix import Matrix_Display
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from skopt.space import Real, Categorical
from skopt import BayesSearchCV
    # This line assigns the variables below to the output of the function load_and_preprocess_data
    # the file path is given at the botton of this page
X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()

    # Logistic Regression
pipeline= Pipeline ([
    ('smotetomek', SMOTETomek(random_state=42)),
    ("LgR", LogisticRegression(max_iter=10000, random_state=42))
    ])

search_space = {
    'LgR__C': Real(1e-4, 1e2, prior='log-uniform'),  # Regularization strength
    'LgR__solver': Categorical(['lbfgs', 'saga']),  # solvers that support l2 penalty and multinomial
}
opt = BayesSearchCV(pipeline, search_space, cv=10, n_iter=20, scoring='f1_macro', random_state=42)
opt.fit(X_train, y_train)
