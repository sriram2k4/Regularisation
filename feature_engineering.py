import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoding(dataset):
    catogorical_column = dataset.select_dtypes(include=["object"]).columns
    print(catogorical_column)

    one_hot_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
    one_hot_encoded = one_hot_encoder.fit_transform(dataset[catogorical_column])
    # print(dataset.info())

    one_hot_encoded = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(catogorical_column)) # Look in this tommorow

    dataset = dataset.drop(catogorical_column, axis=1)

    dataset = pd.concat([dataset,one_hot_encoded],axis=1)

    dataset.to_csv("preprocessed_data.csv")

