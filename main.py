# Created by Antonio Di Mariano (antonio.dimariano@gmail.com) at 22/11/2019
from classes.EDA import EDA
from classes.LinearAlgorithms import LineaRegression_Prediction

_std_label_to_study = 'SalePrice'

if __name__ == "__main__":
    # first load test and train data
    eda = EDA(test_data_csv='test.csv', train_data_csv='train.csv')
    # features's categories. I divide the categories in quantitatives and qualitatives
    train_categorized_variables = eda.extract_quantitative_and_qualitative_variables(data_set=eda.train_data)
    test_categorized_variables = eda.extract_quantitative_and_qualitative_variables(data_set=eda.test_data)

    # HEre is you want to have a nice output :)
    # print("Quantitative's features:",categorized_variables.get('quantitative'))
    # print("Qualitative's features:", categorized_variables.get('qualitative'))
    # see the correlation with quantitative features
    # print("---------------------------------")


    eda.print_bivariants_correlation_with_quantitative_features(label=_std_label_to_study, data_set=eda.train_data,
                                                                quantitative_list=train_categorized_variables.get(
                                                                    'quantitative'))

    # Plot the SalePrice #histogram and normal probability plot
    eda.histogram_and_normal_probability(data_set=eda.train_data, label=_std_label_to_study)

    # Correlation matrix
    correlation_mat_df = eda.correlation_matrix(data_set=eda.train_data)
    print(correlation_mat_df)

    # plot a detailed correlation matrix
    eda.detailed_correlation_matrix(data_set=eda.train_data,matrix=correlation_mat_df,feature_to_zoom=_std_label_to_study,show=True)

    #
    # Here for each of the qualitative and quantitative features a plot is created.
    #
    """
        for feature in categorized_variables['qualitative']:
        eda.plot_qualitative_relationship_against_lable(data_set=eda.train_data, feature=str(feature), label=_std_label_to_study)

    for feature in categorized_variables['quantitative']:
        eda.plot_quantitative_relationship_against_lable(data_set=eda.train_data, feature=str(feature),label=_std_label_to_study)
    
    """

    # Value Missing and Imputing

    eda.get_total_and_percentage_of_missing_data(title='Quantitative features',
                                                 dataset=eda.train_data[train_categorized_variables.get('quantitative')])
    eda.get_total_and_percentage_of_missing_data(title='Qualitative features',
                                                 dataset=eda.train_data[train_categorized_variables.get('qualitative')])

    # now imputing missing values with scalar and values and then check again

    ## For train data
    for quantitative in train_categorized_variables['quantitative']:
        eda.imputing_missing_value_with_value(dataset=eda.train_data, feature_with_missing_value=quantitative)

    for qualitative in train_categorized_variables['qualitative']:
        eda.imputing_missing_value_with_value(dataset=eda.train_data, feature_with_missing_value=qualitative,
                                              value_to_fill='NA')


    ## For test data

    eda.get_total_and_percentage_of_missing_data(title='Quantitative features',
                                                 dataset=eda.test_data[train_categorized_variables.get('quantitative')])
    eda.get_total_and_percentage_of_missing_data(title='Qualitative features',
                                                 dataset=eda.test_data[train_categorized_variables.get('qualitative')])





    ## here we can plot a grid to visualize how each qualitative features is distributed

    eda.visualize_the_distribution_grid_of_numerical_features(dataset=eda.train_data,label=_std_label_to_study,quantitative_list=train_categorized_variables.get('quantitative'))

    ## here we can plot a grid to visualize how each quantitative features is distributed
    eda.visualize_the_scatter_plot_grid_of_numerical_features(dataset=eda.train_data, label=_std_label_to_study,quantitative_list=train_categorized_variables.get('quantitative'))

    """
    here we can plot a scatter plot with automatic regression btw SalePrice and one of the feature with the highest correlation
    We see how data are distributed against a linear regression
    """

    eda.plot_scatter_plot_with_automatic_regression(label_a='SalePrice', label_b='GrLivArea', data=eda.train_data)

    eda.histogram(data=eda.train_data[_std_label_to_study])
    """
     we see that the distribution is skewed. 
    To make the distribution more symmetric, we can try taking its logarithm
    """
    eda.histogram(data=eda.train_data[_std_label_to_study],log_data=True)

    """
    Dummification. 
    I selected all the qualitative features 
    
    """
    eda.dummification(data=eda.train_data,qualitative_list=train_categorized_variables.get('qualitative'))

    list_of_added_features = eda.features_engineering_against_a_dataset(data=eda.train_data)
    print("Added features:",list_of_added_features)
    for feature in list_of_added_features:
        print(eda.train_data[feature].head())

    # This dict has the ratio I want to add
    ratio_to_add_dict = {
        'Total_num_of_bathrooms': {
                'against':'TotalArea',
                'col_name':'Total_num_of_bathrooms_vs_TotalArea'
        },
        'Total_num_of_halfbathrooms': {
            'against': 'TotalArea',
            'col_name': 'Total_num_of_halfbathrooms_vs_TotalArea'
        },
        'GarageCars': {
            'against': 'TotalArea',
            'col_name': 'GarageCars_vs_TotalArea'
        }
     }

    for ratio in ratio_to_add_dict:

        eda.adding_ratio_between_columns(data=eda.train_data, feature_a=ratio,
                                         feature_b=ratio_to_add_dict[ratio].get('against'),
                                         column_to_add=ratio_to_add_dict[ratio].get('col_name'))



    features_to_model = list(ratio_to_add_dict.keys()) +list_of_added_features
    print("Features:",features_to_model)
    ## Model
    linear_reg_with_train_data = LineaRegression_Prediction(dataset=eda.train_data, features_list=features_to_model, label=_std_label_to_study)
    linear_reg_with_train_data.build_training_set()
    linear_reg_with_train_data.fit_model(set_x=linear_reg_with_train_data.training_set_x, set_y=linear_reg_with_train_data.training_set_y)
    linear_reg_with_train_data.make_prediction(validation_data=linear_reg_with_train_data.validation_data_x)
    # now with all dataset
    X = eda.train_data[features_to_model]
    y = eda.train_data[_std_label_to_study]
    linear_reg_with_train_all_data = LineaRegression_Prediction(dataset=eda.train_data, features_list=features_to_model, label=_std_label_to_study)

    linear_reg_with_train_all_data.fit_model(set_x=X,set_y=y)
    linear_reg_with_train_data.make_prediction(validation_data=linear_reg_with_train_data.validation_data_x)
    linear_reg_with_train_data.print_out(test_data=eda.test_data)
     #Make prediciton on test data


