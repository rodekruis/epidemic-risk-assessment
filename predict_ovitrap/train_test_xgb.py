import pandas as pd
from prepare_dataset import prepare_dataset_for_training
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def train_test_xgb(dataset='data/dataset_for_analysis.csv',
                   finetuning=False):
    """ train and test a xgb model
    1) define target and predictors, prepare dataset
    2) OPTIONAL: hyper-parameter tuning
    3) train and test model, save CV-scores
    4) print scores and feature importance
    """

    # 1) define target and predictors, prepare dataset

    # define labeld of predictors and target
    X_labels_base = ['MODIS_006_MOD11A1_LST_Day_1km',
                     'MODIS_006_MOD11A1_LST_Night_1km',
                     'MODIS_006_MYD13A1_EVI',
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
    if not os.path.exists(dataset):
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
        X_train, X_test, y_train, y_test = train_test_split(df[X_labels], df[y_label], test_size=0.1)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        # define grid of possible hyper-parameter values
        gridsearch_params = [
            (max_depth, learning_rate, n_estimators)
            for max_depth in range(1, 10)
            for learning_rate in [0.05, 0.1, 0.2]
            for n_estimators in range(100, 100, 100)
        ]
        # names of hyper-parameters that we are going to tune
        param_names = ['max_depth', 'learning_rate', 'n_estimators']
        # initial hyper-parameters
        params = {
            'max_depth': 6,
            'min_child_weight': 7,
            'n_estimators': 1000,
            'eta': .3,
            'subsample': 1,
            'colsample_bytree': 1,
            'subsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree'
        }
        # loop over grid, find best hyper-parameters
        min_rmse = float("Inf")
        best_params = None
        for params_grid in tqdm(gridsearch_params):
            # Update our parameters
            for name, value in zip(param_names, params_grid):
                params[name] = value
            n_rounds = int(2000/(params['learning_rate']/0.05))
            # Run CV
            cv_results = xgb.cv(
                params,
                dtrain,
                num_boost_round=n_rounds,
                seed=42,
                nfold=5
            )
            # Update best RMSE
            mean_rmse = cv_results['test-rmse-mean'].min()
            boost_rounds = cv_results['test-rmse-mean'].values.argmin()
            print("\trmse {} for {} rounds".format(mean_rmse, boost_rounds))
            if mean_rmse < min_rmse:
                min_rmse = mean_rmse
                best_params = params
        # print results
        print('params:', param_names)
        print("best params: {}, rmse: {}".format(best_params, min_rmse))
    else:
        # used fixed hyper-parameters
        best_params = {'max_depth': 9, 'min_child_weight': 7, 'n_estimators': 1000, 'eta': 0.3, 'subsample': 1,
                       'colsample_bytree': 1, 'subsample_bytree': 0.8, 'objective': 'reg:squarederror',
                       'eval_metric': 'rmse', 'booster': 'gbtree', 'learning_rate': 0.2}

    # 3) train and test model, save CV-scores

    # initialize performance metrics & feature importance
    mae, mse, evs, r2s = 0., 0., 0., 0.
    feature_importances_sum = [0. for i in range(len(X_labels))]
    # define number of folds in cross-validation
    n_folds = 10

    # for each fold: random split dataset, train and test xgb model
    for i in tqdm(range(n_folds)):
        X_train, X_test, y_train, y_test = train_test_split(df[X_labels], df[y_label], test_size=0.1)
        xg_reg = xgb.XGBRegressor(**best_params)
        xg_reg.fit(X_train, y_train)
        y_pred = xg_reg.predict(X_test)
        mae += mean_absolute_error(y_test, y_pred)
        mse += mean_squared_error(y_test, y_pred)
        evs += explained_variance_score(y_test, y_pred)
        r2s += r2_score(y_test, y_pred)
        for i, fi in enumerate(xg_reg.feature_importances_):
            feature_importances_sum[i] += fi

    # 4) print scores and feature importance

    # print average performance & feature importance
    print("cv-performance: MAE {}, MSE {}, EVS {}, R2S {}".format(mae/n_folds, mse/n_folds, evs/n_folds, r2s/n_folds))

    feature_importances = sorted(zip(X_labels, [f/n_folds for f in feature_importances_sum]),
                                 key=lambda t: t[1], reverse=True)
    print("best 10 features:")
    for name, importance in feature_importances[:10]:
        print(name, importance)


if __name__ == "__main__":
    train_test_xgb()

