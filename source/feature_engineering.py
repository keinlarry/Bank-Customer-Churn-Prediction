from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder_gender = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.encoder_country = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    def fit(self, X, y=None):
        self.encoder_gender.fit(X[['gender']])
        self.encoder_country.fit(X[['country']])
        return self

    def transform(self, X):
        X_clean = X.copy()
        if 'customer_id' in X_clean.columns:
            X_clean.drop(columns=['customer_id'], inplace=True)
        X_clean['gender'] = self.encoder_gender.transform(X_clean[['gender']])
        X_clean['country'] = self.encoder_country.transform(X_clean[['country']])
        X_clean['balance_salary_ratio'] = X_clean['balance'] / (X_clean['estimated_salary'] + 1e-8)
        X_clean['balance_a_zero'] = (X_clean['balance'] == 0).astype(int)
        return X_clean