import pandas as pd
from sklearn.preprocessing import StandardScaler
import dateutil.relativedelta
import numpy as np
import joblib


def normalize_features(df, feature_names):
    """ normalize a set of features with standard scaler
    """
    dfn = df.copy()
    features = dfn[feature_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    dfn[feature_names] = features
    return dfn, scaler


def inverse_transform(df, original_file_name, feature_names):
    """ transform a set of normalized features into original values
    """
    dft = df.copy()
    scaler = joblib.load(original_file_name.partition('.')[0]+'_scaler.save')
    features = dft[feature_names]
    features = scaler.inverse_transform(features.values)
    dft[feature_names] = features
    return dft

def prepare_dataset_for_training(ovitrap_data,
                                 weather_data,
                                 which_weather_data,
                                 filename='',
                                 n_time_steps=3):
    """ prepare dataset for model training and testing
    1) combine data of ovitraps (target) and weather (predictors)
    2) clean and normalize target and predictors
    3) create a dataframe with predictors at different previous time steps, e.g.
    ovitrap_0 | rainfall_0 | rainfall_1 | rainfall_2 ...
    4) save final dataframe and normalization scheme (scaler)
    """

    # 1) combine data of ovitraps (target) and weather (predictors)

    df_ovitrap = pd.read_csv(ovitrap_data, index_col=[0, 1]).sort_index()
    df_weather = pd.read_csv(weather_data, index_col=[0, 1]).sort_index()
    dataset = pd.concat([df_ovitrap, df_weather], axis=1)

    # 2) clean and normalize target and predictors

    dataset = dataset.reset_index()
    dataset.date = pd.to_datetime(dataset.date)
    y_label = ['mean_ovi']
    X_labels = which_weather_data
    dataset.loc[(dataset['mean_ovi'] < 0.) & (dataset['mean_ovi'] > 100.), 'mean_ovi'] = np.nan

    # 3) create a dataframe with predictors at different previous time steps

    # define labels of predictors at different time steps
    X_labels_time = []
    for ts in range(n_time_steps):
        X_labels_time.append([x + '_' + str(ts) for x in X_labels])

    # prepare final dataframe (dates, admin levels)
    adm_levels = dataset.adm_level.unique()
    months = pd.date_range(start='1/1/2012', end='12/1/2019', freq='MS')
    df_final = pd.DataFrame(index=pd.MultiIndex.from_product([adm_levels, months],
                                                             names=['adm_level', 'date']),
                            columns=y_label + ['count_ovi'] + X_labels_time[0] + X_labels_time[1] + X_labels_time[2])

    # loop over all dates and assign values
    for ix, row in dataset.iterrows():
        date_start = pd.to_datetime(row['date'])
        for t in range(0, n_time_steps):
            date_old = date_start - dateutil.relativedelta.relativedelta(months=t)
            try:
                data_old = dataset[(dataset.date == date_old) & (dataset.adm_level == row['adm_level'])]
                for col_num, col_name in enumerate(X_labels_time[t]):
                    df_final.at[(row['adm_level'], date_start), col_name] = data_old[X_labels[col_num]].values[0]
            except:
                continue
        df_final.at[(row['adm_level'], date_start), 'mean_ovi'] = row['mean_ovi']
        df_final.at[(row['adm_level'], date_start), 'count_ovi'] = row['count_ovi']

    # 4) normalize, save final dataframe and scaler

    # normalize
    all_labels = y_label
    for ts in range(n_time_steps):
        all_labels = all_labels + X_labels_time[ts]
    df_final, scaler = normalize_features(df_final, all_labels)
    # save final dataframe
    df_final.to_csv(filename)
    # save scaler (normalization scheme)
    joblib.dump(scaler, filename.partition('.')[0]+'_scaler.save')

