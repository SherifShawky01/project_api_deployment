# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, y):
        y = np.array(y).ravel()
        self.encoder.fit(y)
        return self

    def transform(self, y):
        y = np.array(y).ravel()
        return self.encoder.transform(y)

    def inverse_transform(self, y):
        y = np.array(y).ravel()
        return self.encoder.inverse_transform(y)