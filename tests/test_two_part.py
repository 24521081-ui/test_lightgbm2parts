"""Kiểm thử hành vi Two-Part, fallback, đệ quy và kiểm tra khóa."""
import joblib
import numpy as np
import pandas as pd
import pytest
from run import load_config, main
from src.feature_engineering import prepare_training_data, next_day_features
from src.forecasting import recursive_forecast, SeasonalNaive, validate_forecasts
from src.reporting import rmsse_scale, evaluate_forecast, training_patterns
from src.training import train_model, TwoPartModel, ConstantModel


def history(values):
    n = len(values)
    return pd.DataFrame({'item_id': pd.Categorical(['A']*n), 'store_id': pd.Categorical(['CA_1']*n),
        'day': np.arange(1,n+1), 'sales': np.asarray(values, dtype='float32'),
        'day_of_week': np.arange(n)%7, 'month': 1, 'event_flag': 0, 'snap': 0.})


def calendar(origin=60):
    return pd.DataFrame({'d': [f'd_{d}' for d in range(origin+1,origin+29)],
        'day_of_week': np.arange(28)%7, 'month': 1, 'event_flag': 0, 'snap_CA': 0})


@pytest.mark.parametrize('values,policy', [(np.zeros(60), 'constant_zero'), (np.ones(60), 'constant_one'), (np.arange(60)%3, 'lightgbm_binary')])
def test_fit_and_fallback(tmp_path, values, policy):
    data = prepare_training_data(history(values))
    config = load_config('configs/config.yaml')
    params = {key: {**config[key], 'n_estimators': 3, 'n_jobs': 1, 'min_child_samples': 1} for key in ['classifier','regressor']}
    model_path = tmp_path/'thư mục mô hình'/'model.joblib'
    bundle, info = train_model(data, params, model_path)
    assert info['classifier_policy'] == policy
    assert info['n_positive_rows'] == (values[28:] > 0).sum()
    p, amount = bundle['model'].predict_components(data['X'])
    assert np.allclose(bundle['model'].predict(data['X']), p*amount)
    loaded = joblib.load(model_path)
    assert np.array_equal(loaded['model'].predict(data['X']), p*amount)
    if policy == 'constant_zero':
        assert np.all(amount == 0)
    if policy == 'constant_one':
        assert np.all(p == 1)
    if policy == 'lightgbm_binary':
        from lightgbm import Booster
        text = (model_path.parent/'classifier.txt').read_text(encoding='utf-8')
        restored = Booster(model_str=text)
        assert np.allclose(restored.predict(data['X']), p)


def test_features_keep_zeros_and_match_predict():
    values = np.arange(60)%4
    data = prepare_training_data(history(values))
    assert len(data['y']) == 32
    assert (data['y'] == 0).any()
    row = calendar().iloc[0]
    X = next_day_features(values[:59][None,:], ['A'], row, 'CA', ['A'])
    expected = data['X'].iloc[-1]
    for name in ['lag_1','lag_7','lag_14','lag_28','rolling_mean_7','rolling_mean_28']:
        assert X[name].iloc[0] == pytest.approx(expected[name])


class EchoLag:
    def predict_components(self, X):
        return np.full(len(X), .5), X.lag_1.to_numpy()


def test_recursion_product_and_seasonal():
    data = prepare_training_data(history(np.arange(60)))
    common = dict(history=np.arange(60)[None,:], items=['A'], store='CA_1', state='CA', calendar=calendar(), origin=60,
        features=data['features'], categories=['A'])
    forecast, components = recursive_forecast(EchoLag(), **common, return_components=True)
    assert np.allclose(forecast.y_pred, 59*.5**np.arange(1,29))
    assert np.allclose(forecast.y_pred, components.probability_positive*components.conditional_units)
    assert common['history'][0,-1] == 59
    baseline = recursive_forecast(SeasonalNaive(), **common, model_name='seasonal_naive')
    assert np.array_equal(baseline.y_pred, np.tile(np.arange(53,60),4))
    validate_forecasts(forecast)
    with pytest.raises(ValueError):
        validate_forecasts(pd.concat([forecast,forecast.iloc[:1]]))
    actual = pd.DataFrame({'item_id':['A'], 'store_id':['CA_1'], **{f'd_{d}':[0.] for d in range(61,89)}})
    result, _, _, _ = evaluate_forecast(forecast, actual, common['history'], ['A'])
    assert result['n_items'] == 1
    with pytest.raises(ValueError):
        evaluate_forecast(forecast, pd.concat([actual,actual]), common['history'], ['A'])


def test_rmsse_and_selection():
    assert rmsse_scale([0,0,1,3,1]) == 4
    assert np.isnan(rmsse_scale([0,0,0]))
    assert np.isnan(rmsse_scale([0,1,1]))
    patterns, selected, component = training_patterns(np.array([[0,0,0,1],[1,2,3,4]]), ['B','A'])
    assert component == 'B' and set(selected) == {'A','B'}


def test_invalid_component_and_final_test_guard():
    with pytest.raises(ValueError):
        TwoPartModel(ConstantModel(2), ConstantModel(1)).predict(pd.DataFrame({'x':[1]}))
    with pytest.raises(SystemExit) as error:
        main(['--mode','test'])
    assert error.value.code == 2


def test_holdout_is_read_after_forecasts_and_cannot_change_predictions(tmp_path, monkeypatch):
    """Thay toàn bộ nhãn holdout không làm thay đổi dự báo đã tạo."""
    import argparse
    import run
    config = load_config('configs/config.yaml')
    for key in ['classifier', 'regressor']:
        config[key].update(n_estimators=3, n_jobs=1)
    calls = []
    future_value = [0.]
    def load_fake(directory, store, end_day, **kwargs):
        calls.append(('read', kwargs.get('start_day',1), end_day))
        start = kwargs.get('start_day',1)
        if start > 1:
            assert sum(event[0] == 'forecast' for event in calls) == 2
        return pd.DataFrame({'item_id':['A'], 'store_id':['CA_1'], 'state_id':['CA'],
            **{f'd_{day}':[float(day%4) if start == 1 else future_value[0]] for day in range(start,end_day+1)}})
    cal = pd.DataFrame({'d':[f'd_{d}' for d in range(1,1914)], 'wm_yr_wk':1,
        'day_of_week':np.arange(1913)%7, 'month':1, 'event_flag':0, 'snap_CA':0, 'snap_TX':0, 'snap_WI':0})
    original_forecast = run.recursive_forecast
    def forecast(*args, **kwargs):
        result = original_forecast(*args, **kwargs)
        calls.append(('forecast',))
        return result
    monkeypatch.setattr(run, 'load_store', load_fake)
    monkeypatch.setattr(run, 'load_calendar', lambda _: cal)
    monkeypatch.setattr(run, 'recursive_forecast', forecast)
    monkeypatch.setattr(run, 'evaluate_and_export', lambda prediction, *args, **kwargs: prediction)
    args = argparse.Namespace(mode='smoke', scope='smoke')
    first = run.run_store(config, args, tmp_path, tmp_path/'results', 'CA_1')
    assert calls[0] == ('read', 1, 1885)
    assert calls[-1] == ('read', 1886, 1913)
    calls.clear()
    future_value[0] = 999999.
    second = run.run_store(config, args, tmp_path, tmp_path/'results', 'CA_1')
    pd.testing.assert_frame_equal(first, second)


def test_train_only_stops_before_forecast_and_holdout(tmp_path, monkeypatch):
    import argparse
    import run
    config = load_config('configs/config.yaml')
    items = [f'I{i:04}' for i in range(3049)]
    reads = []
    def fake_load(directory, store, end_day, **kwargs):
        reads.append((end_day, kwargs.get('sample_items')))
        assert end_day == 1885 and kwargs.get('sample_items') is None
        return pd.DataFrame({'item_id':items, 'store_id':'CA_1', 'state_id':'CA'})
    monkeypatch.setattr(run, 'load_store', fake_load)
    monkeypatch.setattr(run, 'load_calendar', lambda _: None)
    monkeypatch.setattr(run, 'prepare_history', lambda *args: history(np.arange(60)%3))
    monkeypatch.setattr(run, 'history_matrix', lambda *args: np.zeros((3049,28)))
    def fake_train(data, parameters, path):
        return {'categories':items}, {'training_time_seconds':0., 'actual_trees':{'classifier':200,'regressor':200}}
    monkeypatch.setattr(run, 'train_model', fake_train)
    def forbidden(*args, **kwargs):
        raise AssertionError('Train-only must not forecast/evaluate/read holdout')
    monkeypatch.setattr(run, 'recursive_forecast', forbidden)
    monkeypatch.setattr(run, 'evaluate_and_export', forbidden)
    meta = run.run_store(config, argparse.Namespace(mode='train',scope='full'), tmp_path, tmp_path/'results','CA_1')
    assert reads == [(1885,None)]
    assert meta['mode'] == 'train' and meta['origin_day'] == 1885
    assert meta['forecast_performed'] is False and meta['evaluation_performed'] is False
    assert list((tmp_path/'results').iterdir()) == [tmp_path/'results/models']
    assert len(list((tmp_path/'results/models').glob('*/metadata.json'))) == 1


def test_train_cli_rejects_smoke_scope_and_resume_other_modes():
    with pytest.raises(SystemExit) as error:
        main(['--mode','train','--scope','smoke'])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(['--mode','validation','--resume-training'])
    assert error.value.code == 2


def test_error_columns_and_excel_keep_precision(tmp_path):
    from openpyxl import load_workbook
    from src.reporting import add_forecast_errors, validation_error_tables, export_validation_workbook
    sources=[]
    for store, adjustment in [('CA_1',0.),('TX_1',.000000000123)]:
        detail=pd.DataFrame({'store_id':store,'item_id':'A','horizon':np.arange(1,29),
            'y_actual':10.,'y_pred_two_part':10.+np.tile([.123456789123,-.234567891234],14)+adjustment,
            'y_pred_seasonal_naive':np.tile([9.,12.],14)})
        detail=add_forecast_errors(detail)
        assert np.array_equal(detail.signed_error_two_part,detail.y_pred_two_part-detail.y_actual)
        assert np.array_equal(detail.absolute_error_two_part,np.abs(detail.y_pred_two_part-detail.y_actual))
        totals,horizons=validation_error_tables(detail)
        row=totals.loc[totals.model.eq('lightgbm_two_part')].iloc[0]
        assert row.MAE == pytest.approx(np.abs(detail.y_pred_two_part-10).mean(),abs=1e-14)
        assert row.MAE > abs(row.mean_signed_error)
        detail_path=tmp_path/f'{store}_detail.csv'; detail.to_csv(detail_path,index=False)
        summary=tmp_path/f'{store}_summary.csv'
        pd.DataFrame([dict(model=model,n_items=1,MAE=1.,RMSE=2.,RMSSE_mean_unweighted=.5,
            rmsse_valid_items=1,rmsse_undefined_items=0) for model in ['lightgbm_two_part','seasonal_naive']]).to_csv(summary,index=False)
        sources.append(dict(store_id=store,actual_vs_predictions=str(detail_path),summary=str(summary)))
    path=export_validation_workbook(sources,tmp_path/'workbook')
    book=load_workbook(path,data_only=True)
    assert {'Guide','Totals_all','Totals_by_store','Errors_by_horizon','Errors_store_horizon'}.issubset(book.sheetnames)
    overall=pd.read_excel(path,sheet_name='Totals_all')
    row=overall.loc[overall.model.eq('lightgbm_two_part')].iloc[0]
    assert row.n_items==2 and row.n_observations==56 and row.total_actual==560
    expected=28*(.123456789123-.234567891234)+28*.000000000123
    assert row.total_difference==pytest.approx(expected,abs=1e-12)
    assert len(pd.read_excel(path,sheet_name='Errors_by_horizon'))==56
    cell=book['Totals_all']['H2']
    assert cell.number_format != 'General'


def test_saved_validation_never_fits_and_reads_holdout_last(tmp_path,monkeypatch):
    import json
    import run
    from src.feature_engineering import BASE_FEATURES
    items=[f'I{i:04}' for i in range(3049)]
    config=load_config('configs/config.yaml')
    source=tmp_path/'trained'; source.mkdir()
    model=TwoPartModel(ConstantModel(0),ConstantModel(0))
    bundle=dict(model=model,features=BASE_FEATURES,categories=items,origin_day=1885,train_start_day=1,fit_start_day=29)
    joblib.dump(bundle,source/'model.joblib')
    joblib.dump(model.classifier,source/'classifier.joblib'); joblib.dump(model.regressor,source/'regressor.joblib')
    metadata=dict(status='completed',mode='train',config=config,scope='full',store_id='CA_1',state_id='CA',
        n_items=3049,origin_day=1885,fit_end_day=1885,features=BASE_FEATURES,categories=items,selected_items=items,training_time_seconds=0.)
    (source/'metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
    inventory=tmp_path/'train.csv'; pd.DataFrame([dict(store_id='CA_1',model_path=str(source/'model.joblib'))]).to_csv(inventory,index=False)
    events=[]
    def forbidden(*args,**kwargs):
        raise AssertionError('Saved Validation must not train')
    monkeypatch.setattr(run,'train_model',forbidden)
    monkeypatch.setattr(run,'prepare_training_data',forbidden)
    monkeypatch.setattr(run,'prepare_history',forbidden)
    cal=pd.DataFrame({'d':[f'd_{i}' for i in range(1886,1914)],'day_of_week':0,'month':1,'event_flag':0,'snap_CA':0})
    monkeypatch.setattr(run,'load_calendar',lambda _:cal)
    def load(directory,store,end_day,**kwargs):
        events.append(('read',kwargs.get('start_day',1),end_day))
        if kwargs.get('start_day',1)==1886:
            assert sum(e[0]=='forecast' for e in events)==2
            return pd.DataFrame({'item_id':items,'store_id':store,**{f'd_{d}':0. for d in range(1886,1914)}})
        assert end_day==1885 and kwargs.get('sample_items') is None
        return pd.DataFrame({'item_id':items,'state_id':'CA','store_id':store})
    monkeypatch.setattr(run,'load_store',load)
    monkeypatch.setattr(run,'history_matrix',lambda *args:np.zeros((3049,1885),dtype='float32'))
    original=run.recursive_forecast
    def forecast(*args,**kwargs):
        result=original(*args,**kwargs); events.append(('forecast',)); return result
    monkeypatch.setattr(run,'recursive_forecast',forecast)
    monkeypatch.setattr(run,'export_validation_workbook',lambda sources,directory:directory/'validation_summary.xlsx')
    def figures(directory,*args):
        directory.mkdir(parents=True,exist_ok=True)
    monkeypatch.setattr('src.reporting.save_two_part_figures',figures)
    run.run_saved_validation(config,tmp_path,tmp_path/'results',['CA_1'],inventory)
    assert events[0]==('read',1,1885) and events[-1]==('read',1886,1913)
    detail_path=list((tmp_path/'results/validation').glob('validation_C0*/actual_vs_predictions.csv'))[0]
    detail=pd.read_csv(detail_path)
    assert len(detail)==3049*28
    assert (detail[['signed_error_two_part','absolute_error_two_part','signed_error_seasonal_naive','absolute_error_seasonal_naive']]==0).all().all()


def test_c1_only_fits_classifier_reuses_regressor_bytes(tmp_path,monkeypatch):
    from lightgbm import LGBMRegressor
    from src.training import train_classifier_reuse_regressor
    data=prepare_training_data(history(np.arange(60)%3))
    cfg=load_config('configs/config.yaml')
    params={key:{**cfg[key],'importance_type':'gain','n_estimators':3,'n_jobs':1,'min_child_samples':1} for key in ['classifier','regressor']}
    source=tmp_path/'c0/model.joblib'
    base,_=train_model(data,params,source)
    def forbidden(*args,**kwargs):
        raise AssertionError('C1 must not fit regressor')
    monkeypatch.setattr(LGBMRegressor,'fit',forbidden)
    c1,info=train_classifier_reuse_regressor(data,{**params['classifier'],'num_leaves':15},source,tmp_path/'c1/model.joblib')
    assert info['regressor_retrained'] is False
    assert (source.parent/'regressor.joblib').read_bytes()==(tmp_path/'c1/regressor.joblib').read_bytes()
    assert c1['model'].regressor.booster_.model_to_string()==base['model'].regressor.booster_.model_to_string()
    assert c1['model'].classifier.get_params()['num_leaves']==15
    assert c1['features']==base['features'] and c1['categories']==base['categories']
