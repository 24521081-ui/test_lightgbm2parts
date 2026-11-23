"""Đối chiếu độc lập CSV đã lưu với M5 thật; không huấn luyện, không mở Test mới."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loader import load_store, load_calendar
from src.forecasting import FORECAST_COLUMNS, validate_forecasts, recursive_forecast
from src.preprocessing import prepare_history


def verify(tag):
    metadata_path = ROOT / 'results/models' / tag / 'metadata.json'
    meta = json.loads(metadata_path.read_text(encoding='utf-8'))
    origin, items, store = meta['origin_day'], meta['selected_items'], meta['store_id']
    directory = ROOT / meta['config']['results_dir'] / meta['split'] / tag
    main = pd.read_csv(directory / 'lightgbm_two_part.csv')
    baseline = pd.read_csv(directory / 'seasonal_naive.csv')
    detail = pd.read_csv(directory / 'actual_vs_predictions.csv')
    summary = pd.read_csv(ROOT / 'results/metrics' / tag / 'summary.csv').set_index('model')
    data_dir = ROOT / meta['config']['data_dir']
    train = load_store(data_dir, store, origin, item_ids=items).set_index('item_id').loc[items]
    actual = load_store(data_dir, store, origin+28, start_day=origin+1, item_ids=items).set_index('item_id').loc[items]
    train_values = train[[f'd_{d}' for d in range(1,origin+1)]].to_numpy(dtype=float)
    truth = actual[[f'd_{d}' for d in range(origin+1,origin+29)]].to_numpy(dtype=float)
    counts = {}
    for frame, name in [(main,'lightgbm_two_part'), (baseline,'seasonal_naive')]:
        assert list(frame.columns) == FORECAST_COLUMNS
        validate_forecasts(frame)
        predicted = frame.pivot(index='item_id',columns='horizon',values='y_pred').loc[items,range(1,29)].to_numpy()
        error = truth-predicted
        scales = []
        for row in train_values:
            positive = np.flatnonzero(row > 0)
            trimmed = row[positive[0]:] if len(positive) else np.array([])
            scale = np.mean(np.diff(trimmed)**2) if len(trimmed)>1 else np.nan
            scales.append(scale if scale>0 else np.nan)
        scales = np.asarray(scales)
        valid = np.isfinite(scales)
        rmsse = np.sqrt(np.mean(error**2,axis=1)[valid]/scales[valid]).mean() if valid.any() else np.nan
        for metric, value in [('MAE',np.abs(error).mean()),('RMSE',np.sqrt((error**2).mean())),('RMSSE_mean_unweighted',rmsse)]:
            assert np.isclose(summary.loc[name,metric],value, equal_nan=True)
        assert summary.loc[name,'rmsse_valid_items'] == valid.sum()
        assert summary.loc[name,'rmsse_undefined_items'] == (~valid).sum()
        counts[name] = len(frame)
    keys = ['item_id','store_id','target_day']
    actual_long = actual.reset_index().melt(id_vars=['item_id','store_id'], value_vars=[f'd_{d}' for d in range(origin+1,origin+29)],var_name='target_day',value_name='truth')
    joined = detail.merge(actual_long,on=keys,validate='one_to_one',how='outer',indicator=True)
    assert joined['_merge'].eq('both').all()
    assert np.array_equal(joined.y_actual,joined.truth)
    assert np.allclose(detail.y_pred_two_part,detail.probability_positive*detail.conditional_units,rtol=1e-12,atol=1e-12)
    for frame,col in [(main,'y_pred_two_part'),(baseline,'y_pred_seasonal_naive')]:
        check = detail.merge(frame[keys+['y_pred']],on=keys,validate='one_to_one')
        assert len(check) == len(detail) and np.allclose(check[col],check.y_pred,rtol=1e-12,atol=1e-12)
    assert detail.probability_positive.between(0,1).all()
    expected_naive = np.tile(train_values[:,-7:],(1,4))
    naive_values = baseline.pivot(index='item_id',columns='horizon',values='y_pred').loc[items,range(1,29)].to_numpy()
    assert np.array_equal(expected_naive,naive_values)
    bundle = joblib.load(meta.get('source_model_path', metadata_path.parent/'model.joblib'))
    # Phát lại toàn bộ 28 bước từ bundle để kiểm tra lưu/nạp và đệ quy.
    calendar = load_calendar(data_dir)
    last_prices = None
    if meta['config']['use_prices']:
        from src.data_loader import load_prices
        history = prepare_history(train.reset_index(),calendar,origin,load_prices(data_dir,store))
        last_prices = history.loc[history.day==origin].set_index('item_id').loc[items,'sell_price'].to_numpy()
    replay, components = recursive_forecast(bundle['model'],train_values,items,store,train.state_id.iloc[0],calendar,
        origin,bundle['features'],bundle['categories'],split=meta['split'],last_prices=last_prices,return_components=True)
    assert np.allclose(replay.y_pred,main.y_pred,rtol=1e-12,atol=1e-12)
    result = {'tag':tag,'data':'real M5','items':len(items),'rows_per_model':counts,'horizons':28,
        'actual_join':'PASS','component_product':'PASS','independent_metrics':'PASS','seasonal_period_7':'PASS','saved_model_28_day_replay':'PASS'}
    destination = ROOT / 'results/metrics' / tag / 'verification.json'
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

def verify_batch(tag):
    """Tự tính lại từ CSV gốc d_1–d_1913, kiểm tra workbook và các file gộp."""
    import hashlib
    from src.feature_engineering import next_day_features
    from src.training import verify_saved_models
    directory = ROOT / 'results/metrics' / tag
    manifest = json.loads((directory/'validation_manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed' and not manifest['models_retrained'] and not manifest['test_performed']
    assert len(manifest['sources'])==len(set(manifest['stores']))==10
    days = [f'd_{d}' for d in range(1,1914)]
    real = {}
    raw = ROOT / manifest['config']['data_dir'] / 'sales_train_evaluation.csv'
    for chunk in pd.read_csv(raw,usecols=['store_id','item_id']+days,chunksize=256,
                             dtype={d:'float32' for d in days}):
        values=chunk[days].to_numpy(dtype=float)
        for (store,item), row in zip(zip(chunk.store_id,chunk.item_id),values):
            train=row[:1885]; positive=np.flatnonzero(train>0)
            trimmed=train[positive[0]:] if len(positive) else np.array([])
            scale=np.mean(np.diff(trimmed)**2) if len(trimmed)>1 else np.nan
            assert (store,item) not in real
            real[(store,item)]=(row[1885:].copy(),train[-28:].copy(),scale if scale>0 else np.nan)
    assert len(real)==30490
    calendar=load_calendar(raw.parent).set_index('d')
    master=Path(manifest['actual_vs_predictions']).parent
    readers={name:pd.read_csv(master/name,chunksize=3049*28,float_precision='round_trip')
             for name in ['lightgbm_two_part.csv','seasonal_naive.csv','actual_vs_predictions.csv']}
    totals=[]; horizons=[]; metrics=[]; pair_count=0
    for source in manifest['sources']:
        store=source['store_id']; path=Path(source['actual_vs_predictions'])
        detail=pd.read_csv(path,float_precision='round_trip')
        assert len(detail)==3049*28 and detail.store_id.eq(store).all()
        assert not detail.duplicated(['item_id','store_id','target_day']).any()
        assert np.allclose(detail.y_pred_two_part,detail.probability_positive*detail.conditional_units,rtol=1e-12,atol=1e-12)
        assert detail.probability_positive.between(0,1).all()
        model_path=Path(source['source_model_path'])
        assert hashlib.sha256(model_path.read_bytes()).hexdigest()==source['source_model_sha256']
        bundle=verify_saved_models(model_path)
        meta=json.loads((model_path.parent/'metadata.json').read_text(encoding='utf-8'))
        assert bundle['origin_day']==meta['fit_end_day']==1885
        assert meta['config']==manifest['config']
        items=meta['selected_items']; assert len(set(items))==3049
        truth=np.stack([real[(store,item)][0] for item in items])
        history=np.stack([real[(store,item)][1] for item in items])
        scales=np.array([real[(store,item)][2] for item in items]); valid=np.isfinite(scales)
        actual=detail.pivot(index='item_id',columns='horizon',values='y_actual').loc[items,range(1,29)].to_numpy()
        assert np.array_equal(actual,truth)
        X=next_day_features(history,items,calendar.loc['d_1886'],meta['state_id'],bundle['categories'])
        probability,amount=bundle['model'].predict_components(X[bundle['features']])
        first=detail.loc[detail.horizon.eq(1)].set_index('item_id').loc[items]
        assert np.allclose(first.probability_positive,probability,rtol=1e-12,atol=1e-12)
        assert np.allclose(first.conditional_units,amount,rtol=1e-12,atol=1e-12)
        saved_scores=pd.read_csv(source['summary'],float_precision='round_trip').set_index('model')
        for suffix,model in [('two_part','lightgbm_two_part'),('seasonal_naive','seasonal_naive')]:
            forecast=pd.read_csv(path.parent/f'{model}.csv',float_precision='round_trip')
            assert list(forecast.columns)==FORECAST_COLUMNS
            validate_forecasts(forecast)
            predicted=forecast.pivot(index='item_id',columns='horizon',values='y_pred').loc[items,range(1,29)].to_numpy()
            from_detail=detail.pivot(index='item_id',columns='horizon',values=f'y_pred_{suffix}').loc[items,range(1,29)].to_numpy()
            assert np.array_equal(predicted,from_detail)
            error=predicted-truth; absolute=np.abs(error)
            for prefix,expected in [('signed_error',error),('absolute_error',absolute)]:
                actual_error=detail.pivot(index='item_id',columns='horizon',values=f'{prefix}_{suffix}').loc[items,range(1,29)].to_numpy()
                assert np.array_equal(actual_error,expected)
            if suffix=='seasonal_naive':
                assert np.array_equal(predicted,np.tile(history[:,-7:],(1,4)))
            scores=np.sqrt(np.mean(error**2,axis=1)[valid]/scales[valid])
            metric=dict(model=model,store_id=store,n_items=3049,MAE=absolute.mean(),RMSE=np.sqrt((error**2).mean()),
                RMSSE_mean_unweighted=scores.mean() if len(scores) else np.nan,
                rmsse_valid_items=int(valid.sum()),rmsse_undefined_items=int((~valid).sum()))
            for key in ['MAE','RMSE','RMSSE_mean_unweighted','rmsse_valid_items','rmsse_undefined_items']:
                assert np.isclose(saved_scores.loc[model,key],metric[key],rtol=1e-12,atol=1e-12,equal_nan=True)
            metrics.append(metric)
            totals.append(dict(model=model,store_id=store,n_items=3049,n_observations=truth.size,
                total_actual=truth.sum(),total_prediction=predicted.sum(),total_difference=predicted.sum()-truth.sum(),
                signed_error_sum=error.sum(),absolute_error_sum=absolute.sum(),mean_signed_error=error.mean(),MAE=absolute.mean()))
            for h in range(28):
                horizons.append(dict(model=model,store_id=store,horizon=h+1,n_items=3049,n_observations=3049,
                    total_actual=truth[:,h].sum(),total_prediction=predicted[:,h].sum(),
                    total_difference=predicted[:,h].sum()-truth[:,h].sum(),signed_error_sum=error[:,h].sum(),
                    absolute_error_sum=absolute[:,h].sum(),mean_signed_error=error[:,h].mean(),MAE=absolute[:,h].mean()))
        for name,reader in readers.items():
            original=pd.read_csv(path.parent/name,float_precision='round_trip')
            pd.testing.assert_frame_equal(next(reader).reset_index(drop=True),original.reset_index(drop=True))
        assert len(list(Path(source['figures']).glob('*.png')))==5
        pair_count+=len(items)
        print('CHECK PASS',store,'3049 items; 85372 rows/model; raw actuals/errors/metrics/source model/master CSV')
    for reader in readers.values():
        assert next(reader,None) is None
    book=pd.read_excel(manifest['workbook'],sheet_name=None)
    by_store=pd.DataFrame(totals); store_horizon=pd.DataFrame(horizons); store_metrics=pd.DataFrame(metrics)
    def compare(sheet,expected,keys):
        columns=list(expected.columns)
        a=book[sheet].sort_values(keys).reset_index(drop=True)[columns]
        b=expected.sort_values(keys).reset_index(drop=True)
        assert len(a)==len(b)
        for col in columns:
            if pd.api.types.is_numeric_dtype(b[col]):
                np.testing.assert_allclose(a[col],b[col],rtol=1e-12,atol=1e-8,equal_nan=True)
            else:
                assert a[col].equals(b[col]),(sheet,col)
    compare('Totals_by_store',by_store,['model','store_id'])
    compare('Errors_store_horizon',store_horizon,['model','store_id','horizon'])
    compare('Metrics_by_store',store_metrics,['model','store_id'])
    columns=['n_items','n_observations','total_actual','total_prediction','signed_error_sum','absolute_error_sum']
    for sheet,frame,keys in [('Totals_all',by_store,['model']),('Errors_by_horizon',store_horizon,['model','horizon'])]:
        expected=frame.groupby(keys)[columns].sum().reset_index()
        expected['total_difference']=expected.total_prediction-expected.total_actual
        expected['mean_signed_error']=expected.signed_error_sum/expected.n_observations
        expected['MAE']=expected.absolute_error_sum/expected.n_observations
        compare(sheet,expected,keys)
    overall=[]
    for model,frame in store_metrics.groupby('model'):
        valid=frame.rmsse_valid_items
        overall.append(dict(model=model,n_items=int(frame.n_items.sum()),MAE=frame.MAE.mean(),
            RMSE=float(np.sqrt((frame.RMSE**2).mean())),RMSSE_mean_unweighted=float((frame.RMSSE_mean_unweighted.fillna(0)*valid).sum()/valid.sum()),
            rmsse_valid_items=int(valid.sum()),rmsse_undefined_items=int(frame.rmsse_undefined_items.sum())))
    compare('Metrics_all',pd.DataFrame(overall),['model'])
    result=dict(status='PASS',batch_tag=tag,stores=10,unique_item_store_pairs=pair_count,rows_per_model=pair_count*28,
        horizons=28,raw_actual_join=True,signed_and_absolute_errors=True,independent_MAE_RMSE_RMSSE=True,
        workbook_totals_and_horizons=True,aggregate_CSVs_match=True,source_model_hashes=True,
        first_day_saved_model_replay=True,models_retrained=False,test_performed=False)
    (directory/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag',help='Tên thư mục một lần chạy đã hoàn thành')
    parser.add_argument('--batch',action='store_true',help='Kiểm tra batch Validation C0, workbook và CSV gộp')
    args=parser.parse_args()
    (verify_batch if args.batch else verify)(args.tag)
