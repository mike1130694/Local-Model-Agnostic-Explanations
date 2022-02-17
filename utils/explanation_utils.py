"""
Created by Nikolas Bayer
"""

import pandas as pd
import numpy as np
import pickle


# noinspection PyTypeChecker
class ExplanationUtils:

    rename_dict = {
        'balance_': 'Account Balance',
        'duration_': 'Duration (months)',
        'history_': 'Loan History',
        'purpose_': 'Purpose',
        'amount_': 'Amount (EUR)',
        'savings_': 'Savings Account',
        'employment_': 'Employment',
        'available_income_': 'Available Income',
        'other_debtors_': 'Guarantee',
        #'status_sex_': 'Personal Status / Sex',
        'residence_': 'Residence Duration',
        'assets_': 'Assets',
        'age_': 'Age (years)',
        'other_loans_': 'Other Loans',
        'housing_': 'Housing',
        'previous_loans_': 'Number of Previous Loans',
        'job_': 'Job',
        'people_liable_': 'Number of dependents',
        'telephone_': 'Telephone',
        #'foreign_worker_': 'Foreign Worker'
    }
    
    rename_dict_full = {
        'balance_': 'Account Balance',
        'duration_': 'Duration (months)',
        'history_': 'Loan History',
        'purpose_': 'Purpose',
        'amount_': 'Amount (EUR)',
        'savings_': 'Savings Account',
        'employment_': 'Employment',
        'available_income_': 'Available Income',
        'other_debtors_': 'Guarantee',
        'status_sex_': 'Personal Status / Sex',
        'residence_': 'Residence Duration',
        'assets_': 'Assets',
        'age_': 'Age (years)',
        'other_loans_': 'Other Loans',
        'housing_': 'Housing',
        'previous_loans_': 'Number of Previous Loans',
        'job_': 'Job',
        'people_liable_': 'Number of dependents',
        'telephone_': 'Telephone',
        'foreign_worker_': 'Foreign Worker'
    }
    
   
    def __init__(self, model, data):
        """
        Utility class to generate explanations
        :param model: tensorflow neural network
        :param data: dataset as dataframe, read from data loader
        """
        self.model = model
        self.data = data
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.preprocessor_le = None
        self.preprocessor = None
        self.X_train_le = None
        self.X_test_le = None

    def data_preprocess(self):
        """
        Data preprocessing steps necessary to apply explanation methods
        * Repeat most of the preprocessing steps from model creation
        * Dice does not need the data to be one hot encoded and normalized already as this is built in its functionality
        * The dice data object will be initialized with the same training data that was used for the model creation and
          not the entire dataset. This way, we still have unseen data points that can be explained
        :return: None
        """
        #
        self.data.rename(columns=self.rename_dict, inplace=True)
        # Create target array and feature matrix
        X = self.data.drop(columns='label')
        y = self.data['label']
        # Create train and test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        self.train_data = X_train.join(pd.DataFrame(y_train, columns=['label']))
        self.test_data = X_test.join(pd.DataFrame(y_test, columns=['label']))
        self.X_train = self.train_data.drop(columns='label')
        self.X_test = self.test_data.drop(columns='label')

    def predict_proba(self, data):
        """
        Predicts probability using NN model
        :param data: in data loader format
        :return: array of probabilities
        """
        with open('Preprocessor.pickle', 'rb') as f:
            preprocessor = pickle.load(f)
        data_values = preprocessor.transform(data)
        return self.model.predict(data_values)

    def exp_preprocess(self):
        """
        This function preprocesses the data to the format required by LIME and Anchors and returns everything needed to
        instantiate an LIME / Anchor object
        :return: Label encoded Training and testing data, feature names, category names, category indices
        """
        # The LIME / Anchor explainer has the following requirements:
        # 1. Input data has to be a numpy array with dimensions (n_training_samples, n_features)
        # 2. Input data has to be label encoded
        # 3. List of feature names in the order of the input data
        # 4. Indices of categorical features as list
        # 5. Dictionary with category names for categorical variables in the following form:
        #    {index_1: [category_1, category_2,...], index_2:....}

        # List of feature names

        feature_names = self.X_train.columns.to_list()

        # Lists of categorical and numerical feature names (needed for preprocessing)
        cat_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.to_list()
        num_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns.to_list()

        # List of indices of categorical features
        cat_indices = [feature_names.index(col) for col in cat_cols]

        # Label encoding and creation of dictionary for categories
        from sklearn.preprocessing import LabelEncoder

        # Copy train and test data
        X_train_le = self.X_train.copy()
        X_test_le = self.X_test.copy()

        cat_names = {}
        for cat_idx in cat_indices:
            # Fit label encoder on training data and transform respective column in training and test data
            le = LabelEncoder()
            le.fit(self.X_train.iloc[:, cat_idx])
            X_train_le.iloc[:, cat_idx] = le.transform(X_train_le.iloc[:, cat_idx])
            X_test_le.iloc[:, cat_idx] = le.transform(X_test_le.iloc[:, cat_idx])
            # Extend dictionary with array of categories and index as key
            cat_names[cat_idx] = le.classes_

        self.X_train_le = X_train_le
        self.X_test_le = X_test_le

        # In order to apply the explainer we have to define a prediction function with the following requirements:
        # 1. The label encoded input data has to be mapped to the input format of the neural network.
        #    This includes one hot encoding of categorical variables (Note that for the resulting nd-array
        #    it does not matter whether the initial data is label encoded) and min max scaling for numerical variables
        # 2. The output of the prediction function has to be like the sklearn implementation of predict_proba, that is
        #    an array of [P(y=0 | X_i), P(y=1 | X_i)] arrays for every observation i.

        # The mapping of the label encoded input data to the data format required of the NN
        # will be done by a column transformer
        from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
        from sklearn.compose import ColumnTransformer

        # Required transformations: MinMaxScaling for numerical columns and one-hot encoding for categorical columns
        self.preprocessor_le = ColumnTransformer(
            transformers=[('num', MinMaxScaler(), num_cols),
                          ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)]
        )

        # Fit the data on label encoded training data. Note that this results in the same
        # scaler used for the training of the NN as the used numerical training data is the same.
        # After being transformed by the preprocessor the data is exactly the same as the data
        # used for training the neural network
        self.preprocessor_le.fit(X_train_le)

        # Implementation of the prediction function
        # Input X = array of data points or single data point in label encoded format
        # Required transformations: MinMaxScaling for numerical columns and onehot encoding for categorical columns
        self.preprocessor_shap = ColumnTransformer(
            transformers=[('num', MinMaxScaler(), num_cols),
                          ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)]
        )

        # Fit the data on label encoded training data. Note that this results in the same
        # scaler used for the training of the NN as the used numerical training data is the same.
        # After being transformed by the preprocessor the data is exactly the same as the data
        # used for training the neural network
        self.preprocessor_shap.fit(self.X_train)
        return X_train_le, X_test_le, feature_names, cat_names, cat_indices

    def preprocess_new_data(self, X_df):
        """
        Function to preprocess new data for LIME and anchor explanations
        :param X_df:
        :return:
        """
        data = X_df.copy()
        # List of feature names
        feature_names = self.X_train.columns.to_list()

        # Lists of categorical and numerical feature names (needed for preprocessing)
        cat_cols = self.X_train.select_dtypes(include=['object', 'category']).columns.to_list()

        # List of indices of categorical features
        cat_indices = [feature_names.index(col) for col in cat_cols]

        # Label encoding
        from sklearn.preprocessing import LabelEncoder

        for cat_idx in cat_indices:
            # Fit label encoder on training data and transform respective column
            le = LabelEncoder()
            le.fit(self.X_train.iloc[:, cat_idx])
            data.iloc[:, cat_idx] = le.transform(data.iloc[:, cat_idx])

        return data

    def get_prediction_function(self, exp_type):
        """
        This function returns the prediction function required to compute explanations with LIME or Anchor
        :param exp_type: 'lime' or 'anchor'
        :return:
        """
        def predict_fn(X):
            # Single data point has dimensions (18,) but (1, 18) is needed for preprocessing
            # Therefore, reshape input to (-1, 18)
            X_df = pd.DataFrame(X.reshape((-1, 18)), columns=self.X_train.columns)
            if exp_type == 'lime':
                # Preprocess input and create nd-array
                X_transformed = self.preprocessor_le.transform(X_df).toarray()
                # model.predict output has shape (n, 1) but (n,) is needed
                prob = self.model.predict(X_transformed).reshape(-1, )
                # Return predictions in correct format, dim = (n, 2), note that the first entry
                # has to represent P(y=0 | X)
                proba_predictions = np.array([1 - prob, prob]).transpose()
                return proba_predictions
            elif exp_type == 'anchor':
                # Preprocess input and create nd-array
                X_transformed = self.preprocessor_le.transform(X_df).toarray()
                result = (self.model.predict(X_transformed) > 0.5).astype('int32').flatten()
                return result
            elif exp_type == 'shap':
                X_transformed = self.preprocessor_shap.transform(X_df).toarray()
                result = self.model.predict(X_transformed).flatten()
                return result
            else:
                print('Invalid explanation type')
        return predict_fn

    def get_anchor_exp(self, explainer, prediction_function, lime_pred_fun, idx, data_source='train', visualize=True,
                       plot_mode=False, **kwargs):
        """
        Computes anchor explanation, displays visualization, and returns anchor explanation object
        :param lime_pred_fun: needed to calculate probability
        :param data_source:
        :param visualize: Visualize anchor exp
        :param explainer: anchor explainer instance
        :param prediction_function: anchor prediction function
        :param idx: index of data point
        :param plot_mode: If true, returns also probability of classification, needed for plot
        :return: anchor_exp
        """
        # Set data variables to train or test data

        if isinstance(data_source, pd.core.frame.DataFrame):
            data = data_source
            data_le = self.preprocess_new_data(data_source)
        elif data_source == 'train':
            data = self.X_train
            data_le = self.X_train_le
        elif data_source == 'test':
            data = self.X_test
            data_le = self.X_test_le
        else:
            print('Wrong data')
            return

        # Compute explanation and print it
        #anchor_exp = explainer.explain_instance(data_le.values[idx], prediction_function, **kwargs)
        anchor_exp = explainer.explain_instance(data_le.loc[idx].values, prediction_function, **kwargs)
        
        

        # Visualize
        if visualize:
            #int_prediction = int(prediction_function(data_le.values[idx]))
            int_prediction = int(prediction_function(data_le.loc[idx].values))
            print('Prediction: {:s}\n'.format(['Approved', 'Rejected'][int_prediction]))
            print('Anchor: {:s}'.format('\nAND '.join(anchor_exp.names())))
            print('Precision: {:.3f}'.format(anchor_exp.precision()))
            print('Coverage: {:.3f}'.format(anchor_exp.coverage()))
            #print('\nInstance:\n', data.iloc[idx])
            print('\nInstance:\n', data.loc[idx])

        if plot_mode:
            # Return probability, needed for plotting
            #proba = lime_pred_fun(data_le.values[idx])[0][1]
            proba = lime_pred_fun(data_le.loc[idx].values)[0][1]
            return anchor_exp, proba
        else:
            return anchor_exp

    def get_dice_exp(self, explainer, idx, data_source='train', visualize=True, show_only_changes=True, **kwargs):
        """
        Computes dice explanation, displays df visualization, and returns dice explanation object
        :param visualize: Visualize function of dice
        :param data_source: data source, can be train test or new df
        :param show_only_changes: Show only changes in output df
        :param explainer: dice explainer instance
        :param idx: index of data point
        :param kwargs: kwargs for dice_exp.generate_counterfactuals(), e.g. total_CFs, desired_class,...
        :return:
        """
        # Set data variables to train or test data
        if isinstance(data_source, pd.core.frame.DataFrame):
            data = data_source.rename(columns=self.rename_dict)
        elif data_source == 'train':
            data = self.X_train
        elif data_source == 'test':
            data = self.X_test
        else:
            print('Wrong data')
            return

        # Calculate dice explanation and show data frame
        #query_instance = data.iloc[idx, :].to_dict()
        query_instance = data.loc[idx].to_dict()
        
    
        
        # Create explanations
        dice_exp = explainer.generate_counterfactuals(query_instance, **kwargs)
        if visualize:
            # Visualize explanations
            dice_exp.visualize_as_dataframe(show_only_changes=show_only_changes)
        return dice_exp

    def get_lime_exp(self, explainer, prediction_function, idx, data_source='train', visualize=True, **kwargs):
        """
        Computes LIME explanation, displays html visualization, and returns LIME explanation object
        :param visualize: Visualize as html output
        :param data_source: data source, can be train test or new df
        :param explainer: LIME explainer instance
        :param prediction_function: LIME prediction function
        :param idx: index of data point
        :param kwargs: kwargs for lime_explainer.explain_instance(), e.g. num_features
        :return:
        """
        # Set data variables to train or test data
        if isinstance(data_source, pd.core.frame.DataFrame):
            data_le = self.preprocess_new_data(data_source)
        elif data_source == 'train':
            data_le = self.X_train_le
        elif data_source == 'test':
            data_le = self.X_test_le
        else:
            print('Wrong data')
            return

        # Compute explanation
        #lime_exp = explainer.explain_instance(data_le.values[idx], prediction_function, **kwargs)
        lime_exp = explainer.explain_instance(data_le.loc[idx].values, prediction_function, **kwargs)
        if visualize:
            lime_exp.show_in_notebook(show_all=False)
        return lime_exp

    def get_shap_exp(self, explainer, idx, data_source='train', visualize=True, **kwargs):

        """
        Computes SHAP explanation, displays html visualization, and returns LIME explanation object
        :param visualize: Visualize as html output
        :param data_source: data source, can be train test or new df
        :param explainer: LIME explainer instance
        :param prediction_function: LIME prediction function
        :param idx: index of data point
        :param kwargs: kwargs for lime_explainer.explain_instance(), e.g. num_features
        :return:
        """
        # Set data variables to train or test data
        if isinstance(data_source, pd.core.frame.DataFrame):
            data = data_source.rename(columns=self.rename_dict)
        elif data_source == 'train':
            data = self.X_train
        elif data_source == 'test':
            data = self.X_test
        else:
            print('Wrong data')
            return

        # Compute explanation
        #shap_values = explainer.shap_values(data.iloc[idx,:], **kwargs)
        shap_values = explainer.shap_values(data.loc[idx], **kwargs)
        
        
        if visualize:
            import shap
            #shap.force_plot(explainer.expected_value, shap_values, self.X_train.iloc[idx, :], matplotlib=True)
            shap.force_plot(explainer.expected_value, shap_values, self.X_train.loc[idx], matplotlib=True)
        return shap_values

    def lime_exp_to_df(self, explanations, merge_data='train'):
        """
        Function to compute dataframe of submodular pick data
        :param explanations:
        :param merge_data:
        :return:
        """
        if merge_data == 'train':
            merge_data = self.train_data
        elif merge_data == 'test':
            merge_data = self.test_data
        else:
            print('merge data must be train or test')
            return

        # Store index for later, as merging resets indices
        merge_data['index'] = merge_data.index
        # Create list of explanations, if single explanation is passed
        if not isinstance(explanations, list):
            explanations = [explanations]
        exp_df_list = []
        # Iterate over each explanation, create and append data frames
        for exp in explanations:
            # Get numerical (int) values out of 'feature_values' dict entry
            num_values = [int(float(element))
                          for element in exp.domain_mapper.__dict__['feature_values']
                          if element != 'True' and element != 'False']
            # Create list iterator
            num_iter = iter(num_values)
            # Get categorical feature values and combine with numerical feature values in list
            # exp.domain_mapper.__dict__['exp_feature_names'] contains list of feature, value pairs in the form of
            # "featurename_value" for cat features and "featurename_" for num features
            feat_list = [feat_value.split('=')[1]  # Splitting cat features return list of length 2
                         if len(feat_value.split('=')) == 2 else next(num_iter)  # Fill with num features
                         for feat_value in exp.domain_mapper.__dict__['exp_feature_names']]
            # Create dataframe and append to list
            exp_df_list.append(pd.DataFrame(data=[feat_list], columns=self.X_train.columns))
        # Concat all df's from list, merge result with merge_data to include label and return
        exp_df = pd.concat(exp_df_list, ignore_index='False')
        return pd.merge(exp_df, merge_data, how='left').set_index('index')

    @staticmethod
    def perturb_instance(instance, data, n_perturbs=5000, max_num_changes=3, min_num_changes=1, random_seed=42):
        """
        Create perturbations for instance for CF prediction task
        :param instance: instance to be perturbed
        :param data: data for perturbation distribution, should be from data loader
        :param n_perturbs: number of perturbed instances in result
        :param max_num_changes: max number of changes for each perturbed instance
        :return: df with perturbed instances
        """

        np.random.seed(random_seed)
        # Dict with all features + values
        value_dict = {}
        for col in data.columns:
            value_dict[col] = data[col].unique()

        result = pd.DataFrame(columns=data.columns)
        for i in range(n_perturbs):
            # Copy original instance
            result_instance = instance.copy()
            # Determine number of changes
            num_changes = np.random.choice(np.arange(min_num_changes, max_num_changes+1))
            # Determine indices of features to be changed
            change_indices = np.random.choice(data.shape[1], size=num_changes, replace=False)
            # Features to be changed
            change_cols = data.columns[change_indices]

            # Change determined features for copy of instance
            for change_col in change_cols:
                # Create array of values for column and delete original value
                values = np.delete(value_dict[change_col],
                                   np.where(value_dict[change_col] == result_instance[change_col]))
                result_instance[change_col] = np.random.choice(values)

            # Append changed instance to result df
            result = result.append(result_instance)
        return result.reset_index(drop=True)
