# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 22/11/2019
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
from sklearn.preprocessing import OneHotEncoder


warnings.filterwarnings("ignore", category=RuntimeWarning)
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


_std_show_option = False
class EDA:

    def __init__(self,test_data_csv,train_data_csv):
        self.test_data = pd.read_csv(test_data_csv)
        self.train_data = pd.read_csv(train_data_csv)



    def extract_quantitative_and_qualitative_variables(self,data_set, drop_id=True):
        """
        This method creates two list. one for the numerical values and one for the categorical values of the
        given data_set
        :param data_set:
        :param drop_id:
        :return:
        """
        # Categorization. divide the data into numerical ("quan") and categorical ("qual") features
        quantitative = list(data_set.loc[:, data_set.dtypes != 'object'].drop('Id', axis=1).columns.values)
        qualitative = list(data_set.loc[:, data_set.dtypes == 'object'].columns.values)
        return {"quantitative": quantitative, "qualitative": qualitative}

    def print_bivariants_correlation_with_quantitative_features(self,data_set, label, quantitative_list):
        """
        This method calculates the correlation in the given dataset against the given label and
        a list of quantitative's attributes
        It prints out the ordered list
        :param data_set:
        :param label:
        :param quantitative_list:
        :return:
        """
        numerical_features = {}
        # remove  the given label from the quantitative's features
        if label in quantitative_list:
            quantitative_list.remove(label)
        for feature in quantitative_list:
            numerical_features[feature] = data_set[label].corr(data_set[feature])

        features_ordered = sorted(numerical_features, key=numerical_features.get, reverse=True)
        for ordered_feature in features_ordered:
            print(
                "Correlation between %s and  %s is %f " % (label, ordered_feature, numerical_features[ordered_feature]))

    def histogram_and_normal_probability(self, data_set, label, fit=norm,show=True):

        """
        this method plots a histogram with a normal probability reference
        It also prints out the skewness and kurtosis
        :param data_set:
        :param label:
        :param fit:
        :param show:
        :return:
        """
        sns.distplot(data_set[label], fit=norm)
        # The SalePrice deviates from normal distribution and is positively biased
        fig = plt.figure()
        res = stats.probplot(data_set[label], plot=plt)
        if show:
            plt.show()
        # skewness and kurtosis
        print("Skewness: %f" % data_set[label].skew())
        print("Kurtosis: %f" % data_set[label].kurt())

    def correlation_matrix(self,data_set, x_size=12, y_size=9, square=True,show=False):
        """
        It plots a correlation heat map matrix
        :param data_set:
        :param x_size:
        :param y_size:
        :param square:
        :param show:
        :return:
        """
        corrmat = data_set.corr()
        plt.subplots(figsize=(x_size, y_size))
        sns.heatmap(corrmat, vmax=.8, square=square)
        if show:
            plt.show()
        return corrmat

    def detailed_correlation_matrix(self,data_set, matrix, feature_to_zoom, number_of_variable_for_heatmap=10, show=False):

        """
        this method plots a heat map of a given feature
        :param data_set:
        :param matrix:
        :param feature_to_zoom:
        :param number_of_variable_for_heatmap:
        :param show:
        :return:
        """
        k = number_of_variable_for_heatmap
        cols = matrix.nlargest(k, feature_to_zoom)[feature_to_zoom].index
        cm = np.corrcoef(data_set[cols].values.T)
        sns.set(font_scale=1.25)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
                         yticklabels=cols.values, xticklabels=cols.values)

        if show:
            plt.show()

    def plot_quantitative_relationship_against_label(self, data_set, feature, label,show=False):
        """
        This method plots the relationship between two quantitative attributes
        :param data_set:
        :param feature:
        :param label:
        :param show:
        :return:
        """
        data = pd.concat([data_set[label], data_set[feature]], axis=1)
        data.plot.scatter(x=feature, y=label, ylim=0.800000);
        if show:
            plt.show()
    def plot_qualitative_relationship_against_label(self, data_set, feature, label,show=False):
        """
        This method plots the relationship between two qualitative attributes

        :param data_set:
        :param feature:
        :param label:
        :param show:
        :return:
        """
        data = pd.concat([data_set[label], data_set[feature]], axis=1)
        f, ax = plt.subplots(figsize=(16, 8))
        fig = sns.boxplot(x=feature, y=label, data=data)
        fig.axis(ymin=0, ymax=800000)
        if show:
            plt.show()


    def get_total_and_percentage_of_missing_data(self,dataset, title, sort_ascending=False, lines_to_print=20):
        """
        Value Missing and Imputing

        :param dataset:
        :param title:
        :param sort_ascending:
        :param lines_to_print:
        :return:
        """
        print("Missing value in ", title)
        total = dataset.isnull().sum().sort_values(ascending=sort_ascending)
        percent = (dataset.isnull().sum() / dataset.isnull().count()).sort_values(ascending=sort_ascending)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data.head(lines_to_print))
        return missing_data

    def imputing_missing_value_with_value(self,dataset, feature_with_missing_value, value_to_fill=0):
        """
        this method fills the given dataset with 0 or the value_to_fill parameter
        :param dataset:
        :param feature_with_missing_value:
        :param value_to_fill:
        :return:
        """
        dataset[feature_with_missing_value].fillna(0, inplace=True)



    def visualize_the_scatter_plot_grid_of_numerical_features(self,dataset, label, quantitative_list,show=False):
        """
        This method creates a grid of scatter plots
        :param dataset:
        :param label:
        :param quantitative_list:
        :param show:
        :return:
        """
        temp = pd.melt(dataset, id_vars=[label], value_vars=quantitative_list)
        grid = sns.FacetGrid(temp, col="variable", col_wrap=4, height=3.0,
                             aspect=1.2, sharex=False, sharey=False)
        grid.map(plt.scatter, "value", 'SalePrice', s=3)
        if show:
            plt.show()

    def plot_scatter_plot_with_automatic_regression(self,label_a,label_b,data,show=False):
        """
        Scatter plot with lineare regression
        :param label_a:
        :param label_b:
        :param data:
        :param show:
        :return:
        """
        sns.regplot(label_a,label_b, data=data)
        if show:
            plt.show()


    def dummification(self,data,qualitative_list):
        """
        This method creates dummy variables
        :param data:
        :param qualitative_list:
        :return:
        """
        dummified = pd.get_dummies(data[qualitative_list])
        data = pd.concat([data,dummified],axis=1)
        return data
    def add_boolean_column_for_a_feature(self,dataset, feature, column_name_to_add):
        """
        Dummification of variables

        :param dataset:
        :param feature:
        :param column_name_to_add:
        :return:
        """
        for feature in dataset[feature]:
            if feature == 0:
                dataset[column_name_to_add] = 0
            else:
                dataset[column_name_to_add] = 1
        return dataset[column_name_to_add]


    def histogram(self,data,log_data=False,bins=25,show=False):
        """
        This method creates a histogram out of the given data.
        If the log_data = True a log(data) is created so as to make
        the distribution more symmetric
        :param data:
        :param log_data:
        :param bins:
        :param show:
        :return:
        """
        if log_data:
            plt.hist(np.log(data), bins=bins)
        else:
            plt.hist(data,bins=bins)
        if show:
            plt.show()


    def features_engineering_against_a_dataset(self,data):
        """
        Naive method to add features to the given dataset
        :param data:
        :return:
        """
        # add a new column to indicate that a house has a basement
        self.add_boolean_column_for_a_feature(data, feature='TotalBsmtSF', column_name_to_add='has_a_basement')
        # print(eda.train_data['has_a_basement'].head())

        # add a new column to indicate that a house has a garage
        self.add_boolean_column_for_a_feature(data, feature='GarageArea', column_name_to_add='has_a_garage')

        # add a new column to indicate that a house has a 2nd floor
        self.add_boolean_column_for_a_feature(data, feature='2ndFlrSF', column_name_to_add='has_a_2_floor')

        # add a new column to indicate that total area of a house

        data['TotalArea'] = data['GrLivArea'] + data['TotalBsmtSF'] + data['GarageArea']

        # plot the distribution of the TotalArea again the label

        #self.plot_scatter_plot_with_automatic_regression(label_a='SalePrice', label_b='TotalArea', data=eda.train_data)

        # I can print out the outliners. 6700 is an observed value
        print('Outliers:', (data['TotalArea'] >= 6700).sum())

        # total full bath number
        data['Total_num_of_bathrooms'] = data['FullBath'] + (data['HalfBath'] * 0.5) +  data['BsmtFullBath'] + (data['BsmtHalfBath'] * 0.5)

        # total half bath

        data['Total_num_of_halfbathrooms'] = data['HalfBath']
        # total number of bathroom above ground
        data['Total_num_of_bathroom_above_ground'] = data['HalfBath'] + data['FullBath']
        data['Total_num_of_rooms_bathrooms_included'] =  data['TotRmsAbvGrd'] + data['Total_num_of_bathrooms']

        return ['has_a_basement','has_a_garage','has_a_2_floor','Total_num_of_bathrooms','Total_num_of_halfbathrooms','TotalArea','Total_num_of_bathroom_above_ground','Total_num_of_rooms_bathrooms_included']

    def adding_ratio_between_columns(self, data, feature_a, feature_b, column_to_add):
        """
        This method add a correlation's ratio between feature_a and feature_b and creates 
        a new column
        :param data: 
        :param feature_a: 
        :param feature_b: 
        :param column_to_add: 
        :return: 
        """

        ratio = data[feature_a].corr(data[feature_b])
        data[column_to_add] = ratio


