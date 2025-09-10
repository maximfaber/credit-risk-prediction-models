
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from skopt.space import Categorical, Integer


from data_prep import load_and_preprocess_data
from ROC_gen import ROC_Generator
from conf_matrix import Matrix_Display
from skopt import BayesSearchCV

# Load the dataset

    # Train-test split
X_train, X_test, y_train, y_test, X, y = load_and_preprocess_data()

    # Logistic Regression
pipeline= Pipeline ([
    ('smotetomek', SMOTETomek(random_state=42)),
    ('RF', RandomForestClassifier(random_state=42))
    ])

search_space = {
    'RF__n_estimators': Integer(50, 500),
    'RF__max_depth': Integer(3, 50),
    'RF__min_samples_split': Integer(2, 20),
    'RF__min_samples_leaf': Integer(1, 20),
    'RF__max_features': Categorical(['sqrt', 'log2', None]),
    'RF__bootstrap': Categorical([True, False])
}

opt = BayesSearchCV(pipeline, search_space, cv=10, n_iter=20, scoring='f1_macro', random_state=42)

opt.fit(X_train, y_train)
