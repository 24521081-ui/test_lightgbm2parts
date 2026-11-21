"""Một thử nghiệm C1: classifier 15 lá, dùng nguyên regressor C0; không vẽ hình/Test."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import copy
import gc
import hashlib
import json
import logging
import shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
from src.data_loader import load_store,load_calendar,check_files
from src.preprocessing import prepare_history,history_matrix
from src.feature_engineering import prepare_training_data
from src.training import train_classifier_reuse_regressor
from src.forecasting import recursive_forecast,export_forecasts,validate_forecasts
from src.reporting import training_patterns,evaluate_and_export,save_json

LOG=logging.getLogger('m5')
C0_TAG='validation_C0_full_10_stores_20261119T103613400752Z'


def diagnostics(frame):
    positive=frame.y_actual>0
    return dict(actual=float(frame.y_actual.sum()),prediction=float(frame.y_pred_two_part.sum()),
        bias_pct=100*(frame.y_pred_two_part.sum()/frame.y_actual.sum()-1),
        p_gap_pp=100*(frame.probability_positive.mean()-positive.mean()),
        mu_bias_pct=100*(frame.loc[positive,'conditional_units'].sum()/frame.loc[positive,'y_actual'].sum()-1))


def main():
    config=yaml.safe_load((ROOT/'configs/config.yaml').read_text(encoding='utf-8-sig'))
    c0=json.loads((ROOT/'results/metrics'/C0_TAG/'validation_manifest.json').read_text(encoding='utf-8'))
    assert config==c0['config'] and config['classifier']['num_leaves']==31 and not config['use_prices']
    assert len(c0['sources'])==10 and c0['rows_per_model']==853720
    c1=copy.deepcopy(config); c1['classifier']['num_leaves']=15
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    batch=f'validation_C1_full_10_stores_{run_id}'
    root=ROOT/config['results_dir']; metrics_root=root/'metrics'/batch; metrics_root.mkdir(parents=True,exist_ok=False)
    aggregate=root/'validation'/batch; aggregate.mkdir(parents=True,exist_ok=False)
    save_json(dict(status='running',configuration='C1',config=c1,c0_batch=C0_TAG),metrics_root/'manifest.json')
    data_dir=ROOT/config['data_dir']; check_files(data_dir,False); calendar=load_calendar(data_dir)
    store_rows=[]; metric_rows=[]; sources=[]; all_pairs=set()
    for source in c0['sources']:
        store=source['store_id']; src_model=Path(source['source_model_path'])
        src_info=json.loads((src_model.parent/'metadata.json').read_text(encoding='utf-8'))
        assert src_info['config']==config and src_info['fit_end_day']==1885 and src_info['n_items']==3049
        src_hash=hashlib.sha256(src_model.read_bytes()).hexdigest()
        assert src_hash==source['source_model_sha256']
        LOG.info('C1 START | %s | classifier leaves=15 | reuse regressor %s',store,src_model.parent)
        wide=load_store(data_dir,store,1885,chunk_size=config['chunk_size'],item_ids=src_info['selected_items'])
        items=wide.item_id.tolist(); assert items==src_info['selected_items'] and len(items)==3049
        history=prepare_history(wide,calendar,1885); matrix=history_matrix(wide,1885)
        _,selected,component=training_patterns(matrix,items)
        del wide
        train=prepare_training_data(history); del history; gc.collect()
        tag=f'validation_C1_full_{store}_{run_id}'
        locations={name:root/name/tag for name in ['validation','models','metrics']}
        parameters={**c1['classifier'],'importance_type':'gain'}
        bundle,info=train_classifier_reuse_regressor(train,parameters,src_model,locations['models']/'model.joblib')
        assert info['n_train_rows']==5661993
        expected_classifier=dict(src_info['effective_model_params']['classifier']); expected_classifier['num_leaves']=15
        assert info['effective_model_params']['classifier']==expected_classifier
        assert info['effective_model_params']['regressor']==src_info['effective_model_params']['regressor']
        del train; gc.collect()
        for name in ['validation','metrics']: locations[name].mkdir(parents=True,exist_ok=False)
        common=dict(history=matrix,items=items,store=store,state=src_info['state_id'],calendar=calendar,origin=1885,
            features=bundle['features'],categories=bundle['categories'],split='validation',horizon=28)
        prediction,components=recursive_forecast(bundle['model'],**common,return_components=True)
        export_forecasts(prediction,locations['validation']/'lightgbm_two_part.csv')
        # Không đọc nhãn Validation cho đến sau đủ 28 bước dự báo C1.
        actual=load_store(data_dir,store,1913,start_day=1886,item_ids=items,chunk_size=config['chunk_size'])
        old=pd.read_csv(source['actual_vs_predictions'],float_precision='round_trip')
        baseline=pd.read_csv(Path(source['actual_vs_predictions']).parent/'seasonal_naive.csv',float_precision='round_trip')
        validate_forecasts(baseline); export_forecasts(baseline,locations['validation']/'seasonal_naive.csv')
        context=dict(config=c1,configuration_name='C1',store_id=store,scope='full',split='validation',origin_day=1885,
            run_id=run_id,batch_tag=batch,illustration_items=selected,component_item=component,save_figures=False,
            mode='validation',status='completed',regressor_retrained=False,source_c0_model_path=str(src_model),
            source_c0_model_sha256=src_hash,forecast_performed=True,evaluation_performed=True)
        scores=evaluate_and_export(prediction,baseline,actual,matrix,items,bundle,info,locations,context,components=components)
        new=pd.read_csv(locations['validation']/'actual_vs_predictions.csv',float_precision='round_trip')
        keys=['split','origin_day','item_id','store_id','target_day','horizon']
        joined=old.merge(new,on=keys,how='outer',suffixes=('_C0','_C1'),indicator=True,validate='one_to_one')
        assert len(joined)==85372 and joined['_merge'].eq('both').all()
        assert np.array_equal(joined.y_actual_C0,joined.y_actual_C1)
        assert np.allclose(new.y_pred_two_part,new.probability_positive*new.conditional_units,rtol=1e-12,atol=1e-12)
        assert np.array_equal(new.signed_error_two_part,new.y_pred_two_part-new.y_actual)
        scale0=pd.read_csv(Path(source['summary']).parent/'lightgbm_two_part_per_item.csv',float_precision='round_trip').set_index('item_id')
        scale1=pd.read_csv(locations['metrics']/'lightgbm_two_part_per_item.csv',float_precision='round_trip').set_index('item_id')
        np.testing.assert_allclose(scale0.loc[items,'scale'],scale1.loc[items,'scale'],rtol=0,atol=0,equal_nan=True)
        row={'store_id':store}
        for label,detail in [('C0',old),('C1',new)]:
            row.update({f'{key}_{label}':value for key,value in diagnostics(detail).items()})
        store_rows.append(row); metric_rows.append(scores[scores.model=='lightgbm_two_part'].iloc[0].to_dict())
        for item in items:
            assert (store,item) not in all_pairs; all_pairs.add((store,item))
        for name in ['lightgbm_two_part.csv','seasonal_naive.csv','actual_vs_predictions.csv']:
            destination=aggregate/name
            with (locations['validation']/name).open('rb') as reader,destination.open('ab') as writer:
                if destination.stat().st_size: reader.readline()
                shutil.copyfileobj(reader,writer)
        sources.append(dict(store_id=store,run_tag=tag,model_path=str((locations['models']/'model.joblib').resolve()),
            regressor_source=info['source_regressor_path'],regressor_sha256=info['source_regressor_sha256'],
            actual_vs_predictions=str((locations['validation']/'actual_vs_predictions.csv').resolve()),summary=str((locations['metrics']/'summary.csv').resolve())))
        LOG.info('C1 COMPLETE | %s | rows=%d | fit %.2fs | trees=%s',store,len(new),info['training_time_seconds'],info['actual_trees'])
        del bundle,common,matrix,prediction,components,actual,old,new,joined,baseline,scale0,scale1
        gc.collect()
    assert len(all_pairs)==30490 and len(sources)==10
    stores=pd.DataFrame(store_rows); stores.to_csv(metrics_root/'comparison_by_store.csv',index=False,encoding='utf-8-sig')
    current=pd.DataFrame(metric_rows); current.to_csv(metrics_root/'C1_metrics_by_store.csv',index=False)
    overview=[]
    for label in ['C0','C1']:
        metrics=pd.read_csv(ROOT/'results/metrics'/C0_TAG/'Metrics_by_store.csv',float_precision='round_trip') if label=='C0' else current
        metrics=metrics[metrics.model=='lightgbm_two_part']; valid=metrics.rmsse_valid_items
        actual=stores[f'actual_{label}'].sum(); predicted=stores[f'prediction_{label}'].sum()
        overview.append(dict(configuration=label,actual=actual,prediction=predicted,total_difference=predicted-actual,
            bias_pct=100*(predicted/actual-1),MAE=float(np.average(metrics.MAE,weights=metrics.n_items)),
            RMSSE_mean_unweighted=float((metrics.RMSSE_mean_unweighted.fillna(0)*valid).sum()/valid.sum())))
    pd.DataFrame(overview).to_csv(metrics_root/'comparison_overall.csv',index=False,encoding='utf-8-sig')
    save_json(dict(status='completed',configuration='C1',config=c1,c0_batch=C0_TAG,rows_per_configuration=853720,
        n_items=30490,stores=10,origin_day=1885,target_days=[1886,1913],same_keys_as_C0=True,
        same_actuals_as_C0=True,same_RMSSE_scales=True,regressors_retrained=False,test_performed=False,
        figures_generated=False,sources=sources,aggregate_predictions=str(aggregate.resolve())),metrics_root/'manifest.json')
    LOG.info('C1 ALL COMPLETE | %s',metrics_root)
    print(pd.DataFrame(overview).to_string(index=False)); print(stores.to_string(index=False))

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,stream=sys.stdout,format='%(asctime)s | %(message)s',datefmt='%H:%M:%S')
    main()
