import pandas as pd
from utils import prepare_dataset_for_training, inverse_transform
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt

def train_test_xgb(dataset='data/dataset_for_analysis_test.csv',
                   finetuning=False,
                   force_data_prep=True):
    """ train and test a xgb model
    1) define target and predictors, prepare dataset
    2) OPTIONAL: hyper-parameter tuning
    3) train and test model, save CV-scores
    4) print scores and feature importance
    """

    # 1) define target and predictors, prepare dataset

    # notes: fucking FLDAS updates after 2 months
    # alternatives
    # precipitation: NASA/GPM_L3/IMERG_V06

    # define labeld of predictors and target
    X_labels_base = ['MODIS_006_MOD11A1_LST_Day_1km',
                     'MODIS_006_MOD11A1_LST_Night_1km',
                     # 'MODIS_006_MYD13A1_EVI',
                     'NASA_FLDAS_NOAH01_C_GL_M_V001_Qair_f_tavg',
                     'NASA_FLDAS_NOAH01_C_GL_M_V001_Rainf_f_tavg',
                     'NASA_FLDAS_NOAH01_C_GL_M_V001_SoilMoi00_10cm_tavg',
                     'NASA_FLDAS_NOAH01_C_GL_M_V001_SoilTemp00_10cm_tavg',
                     'NASA_FLDAS_NOAH01_C_GL_M_V001_Wind_f_tavg',
                     'NASA_FLDAS_NOAH01_C_GL_M_V001_Tair_f_tavg',
                     'JAXA_GPM_L3_GSMaP_v6_operational_hourlyPrecipRateGC']
    y_label = 'mean_ovi'
    # define number of time steps to use (past observations)
    n_time_steps = 3
    # define labels of predictors at different time steps
    X_labels = []
    for ts in range(n_time_steps):
        X_labels += [x + '_' + str(ts) for x in X_labels_base]

    # prepare dataset
    if not os.path.exists(dataset) or force_data_prep:
        print('dataset not found, preparing from raw data (this might take a wile)')
        # define input data path
        ovitrap_data = 'data/ovitrap_data_month_adm2.csv'
        weather_data = 'data/merged_adm2.csv'
        prepare_dataset_for_training(ovitrap_data,
                                     weather_data,
                                     X_labels_base,
                                     filename=dataset,
                                     n_time_steps=n_time_steps)
    df = pd.read_csv(dataset)
    df = df.reset_index()
    df.date = pd.to_datetime(df.date)
    # save original dataset for later (inference)
    df_uncut = df.copy()
    df_uncut = df_uncut.dropna(subset=X_labels)
    df = df.dropna(subset=[y_label])

    # QA cuts
    # removing a few data points based on poor correlation with observed "good" predictors
    df = df[(df.date.dt.year >= 2013) & (df.date.dt.year <= 2017)]
    bad_provinces = ['Nueva Vizcaya', 'Surigao del Norte', 'Sarangani', 'Siquijor', 'Dinagat Islands', 'Isabela',
                     'Capiz', 'South Cotabato', 'Maguindanao', 'Biliran', 'Davao Occidental', 'Quirino', 'Guimaras',
                     'Aurora']
    df = df[~df.adm_level.isin(bad_provinces)]
    df = df[df.count_ovi > 5]

    # 2) OPTIONAL: hyper-parameter tuning

    if finetuning:
        # define training data
        dtrain = xgb.DMatrix(data=df[X_labels], label=df[y_label])
        # define grid of possible hyper-parameter values
        gridsearch_params = [
            {'max_depth': max_depth,
             'learning_rate': learning_rate,
             'n_estimators': n_estimators,
             'min_split_loss': min_split_loss,
             'subsample': subsample,
             'colsample_bytree': colsample_bytree,
             'reg_lambda': reg_lambda,
             'reg_alpha': reg_alpha,
             'booster': booster}
            for max_depth in [2, 5, 10, 20]
            for learning_rate in [0.05, 0.1, 0.2]
            for n_estimators in [1000, 2000, 5000]
            for min_split_loss in [0., 0.1, 1.]
            for subsample in [0.5, 0.8, 1.]
            for colsample_bytree in [0.3, 0.6, 1.]
            for reg_lambda in [1, 1.5, 2]
            for reg_alpha in [0., 0.1, 1.]
            for booster in ['gbtree']
        ]
        # fixed hyper-parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'num_boost_round': 1000
        }
        # loop over grid, find best hyper-parameters
        min_rmse = float("Inf")
        best_params = None
        for params_grid in tqdm(gridsearch_params):
            # Update parameters
            for name, value in params_grid.items():
                params[name] = value
            # Run CV
            cv_results = xgb.cv(dtrain=dtrain, params=params, num_boost_round=params['num_boost_round'],
                                early_stopping_rounds=50, seed=42, nfold=5, as_pandas=True)
            # Update best MAE
            mean_rmse = cv_results['test-rmse-mean'].min()
            boost_rounds = cv_results['test-rmse-mean'].values.argmin()+1
            print("\trmse {} for {} / {} rounds".format(mean_rmse, boost_rounds, len(cv_results)))
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_params = params
        # print results
        print("best params: {}".format(best_params))
        print("min rmse: {}".format(min_rmse))
    else:
        # used fixed hyper-parameters
        best_params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'booster': 'gbtree',
                       'num_boost_round': 1000, 'max_depth': 20, 'learning_rate': 0.2,
                       'n_estimators': 5000, 'subsample': 1.0, 'colsample_bytree': 1.0,
                        'reg_lambda': 2, 'reg_alpha': 1.0, 'min_split_loss': 0.}


    # 3) train and test model, save CV-scores

    nfolds = 5
    X, y = df[X_labels], df[y_label]
    dmdata = xgb.DMatrix(data=X, label=y)
    cv_results = xgb.cv(dtrain=dmdata, params=best_params, nfold=nfolds, metrics=["rmse", "mae"],
                        num_boost_round=best_params['num_boost_round'], early_stopping_rounds=50,
                        as_pandas=True, seed=42)
    boost_rounds = len(cv_results)

    # 4) print scores and feature importance

    # print average performance & feature importance
    print("cv-performance: MAE {}, RMSE {}".format(cv_results["test-mae-mean"].tail(1).values[0],
                                                   cv_results["test-rmse-mean"].tail(1).values[0]))

    xg_reg = xgb.train(dtrain=dmdata, params=best_params, num_boost_round=boost_rounds)

    dpredict = xgb.DMatrix(data=df_uncut[X_labels], label=df_uncut[y_label])
    predictions = xg_reg.predict(dpredict)
    df_pred = df_uncut.copy()
    df_pred[y_label] = predictions
    df_pred = inverse_transform(df_pred, dataset, [y_label] + X_labels)
    df_pred.to_csv('output/dataset_predictions.csv')
    xg_reg.save_model('models/best_model.json')

    df_uncut = df_uncut.dropna(subset=[y_label])
    dpredict = xgb.DMatrix(data=df_uncut[X_labels], label=df_uncut[y_label])
    predictions = xg_reg.predict(dpredict)
    print('R2 train', r2_score(df_uncut[y_label].values, predictions))

    xgb.plot_importance(xg_reg)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_test_xgb()

