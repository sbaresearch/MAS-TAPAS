import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler

def create_random_equal_dfs(df, df_size, num_dfs=4, seed=42):
    if len(df) < df_size * num_dfs:
        raise ValueError("Not enough rows in the original dataframe to create the specified number of dataframes of the given size.")

    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return tuple(df_shuffled[i*df_size:(i+1)*df_size].reset_index(drop=True) for i in range(num_dfs))

def fit_transformer(X, categorical_encoding='one-hot', numeric_encoding='standard'):
    """
    Fit a ColumnTransformer on a single dataset.
    """
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Numeric transformer
    if numeric_encoding == 'standard':
        num_transformer = StandardScaler()
    elif numeric_encoding == 'minmax':
        num_transformer = MinMaxScaler()
    elif numeric_encoding == 'passthrough':
        num_transformer = 'passthrough'
    else:
        raise ValueError("numeric_encoding must be 'standard', 'minmax', or 'passthrough'.")

    # Categorical transformer
    if categorical_encoding == 'one-hot':
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif categorical_encoding == 'ordinal':
        cat_transformer = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    elif categorical_encoding == 'passthrough':
        cat_transformer = 'passthrough'
    else:
        raise ValueError("categorical_encoding must be 'one-hot', 'ordinal', or 'passthrough'.")

    full_transformer = ColumnTransformer([
        ('numerical', num_transformer, numerical_cols),
        ('categorical', cat_transformer, categorical_cols)
    ], remainder='drop')

    full_transformer.fit(X)

    # Feature names
    feature_names = numerical_cols[:]
    if categorical_encoding == 'one-hot' and categorical_cols:
        encoder = full_transformer.named_transformers_['categorical']
        for i, feature in enumerate(categorical_cols):
            categories = encoder.categories_[i]
            for cat in categories:
                feature_names.append(f"{feature}_{cat}")
    else:
        feature_names.extend(categorical_cols)

    return full_transformer, feature_names


class TabularPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-style tabular preprocessor for multiple datasets.
    Supports numeric and categorical encoding options and fit_target selection.
    """

    def __init__(self, fit_target='synth', categorical_encoding='one-hot', numeric_encoding='standard'):
        if fit_target not in ['synth', 'ref']:
            raise ValueError("fit_target must be 'synth' or 'ref'.")
        self.fit_target = fit_target
        self.categorical_encoding = categorical_encoding
        self.numeric_encoding = numeric_encoding

    def fit(self, mem, non_mem, synth, ref=None, y=None):
        # Choose dataset to fit on
        if self.fit_target == 'synth':
            fit_data = synth
        else:  # self.fit_target == 'ref'
            if ref is None:
                raise ValueError("ref dataset must be provided when fit_target='ref'.")
            fit_data = ref

        self.transformer_, self.feature_names_out_ = fit_transformer(
            fit_data,
            categorical_encoding=self.categorical_encoding,
            numeric_encoding=self.numeric_encoding
        )
        return self

    def transform(self, mem, non_mem, synth, ref=None):
        # Transform all datasets; None for ref if not provided
        datasets = [mem, non_mem, synth]
        transformed = [self._transform_single(data) for data in datasets]

        if ref is not None:
            transformed.append(self._transform_single(ref))
        else:
            transformed.append(None)

        return tuple(transformed) + (self.transformer_,)

    def _transform_single(self, X):
        transformed_data = self.transformer_.transform(X)
        # Ensure consistent number of columns
        if transformed_data.shape[1] != len(self.feature_names_out_):
            padding = np.zeros((transformed_data.shape[0], len(self.feature_names_out_) - transformed_data.shape[1]))
            transformed_data = np.hstack([transformed_data, padding])
        return transformed_data

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
