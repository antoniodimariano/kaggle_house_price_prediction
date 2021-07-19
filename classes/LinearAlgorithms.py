# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 22/11/2019
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import pandas as pd


class LineaRegression_Prediction:

    def __init__(self,dataset,features_list,label):

        self._reg = LinearRegression()
        self.dataset = dataset
        self.features_list = features_list
        self.label = label


    def build_training_set(self):
        self.training_set_x, self.validation_data_x, self.training_set_y, self.validation_data_y = train_test_split(
            self.dataset[self.features_list], self.dataset[self.label], test_size=0.2, random_state=5566
        )

    def fit_model(self,set_x,set_y):
        self._reg.fit(set_x,set_y)
        print("Model fitted with training sets")

    def make_prediction(self,validation_data):
        self.prediction = self._reg.predict(validation_data)
        self.rmsle = self.do_rmsle()


    def do_rmsle(self):
        """
         make predictions on validation data
        :param prediction:
        :param y:
        :return:
        """
        return np.sqrt(mean_squared_error(np.log(self.prediction), np.log(self.validation_data_y)))


    def print_out(self,test_data):
        output = pd.DataFrame()
        output[self.label] = self.prediction
        output['Id'] = test_data['Id']
        #print("Prediction:",output.head(20))
        output.to_csv('linear_regression_prediction.csv', index=False)
        print("File created ")

