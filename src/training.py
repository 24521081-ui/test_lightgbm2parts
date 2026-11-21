"""4. Huấn luyện một cặp Two-Part cho mỗi cửa hàng trên Train."""
import logging
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

LOGGER = logging.getLogger('m5')

class ConstantModel:
    """Fallback khi Train chỉ có một lớp hoặc không có ngày bán dương."""
    def __init__(self, value):
        self.value = float(value)
    def predict(self, X):
        return np.full(len(X), self.value)
    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1-self.value), self.predict(X)])
    def get_params(self):
        return {'constant': self.value}

class TwoPartModel:
    def __init__(self, classifier, regressor):
        self.classifier, self.regressor = classifier, regressor
    def predict_components(self, X):
        p = np.asarray(self.classifier.predict_proba(X)[:, 1], dtype=float)
        amount = np.asarray(self.regressor.predict(X), dtype=float)
        if (p.shape != (len(X),) or amount.shape != (len(X),)
                or not np.isfinite(p).all() or not np.isfinite(amount).all()
                or (p < 0).any() or (p > 1).any()):
            raise ValueError('Thành phần Two-Part sai kích thước hoặc không hợp lệ.')
        return p, np.maximum(amount, 0)
    def predict(self, X):
        p, amount = self.predict_components(X)
        return p * amount

def verify_saved_models(model_path, expected=None):
    """Nạp và so cấu trúc/model text; không gọi predict hoặc đọc holdout."""
    model_path = Path(model_path)
    loaded = joblib.load(model_path)
    if expected is not None:
        for name in ['features', 'categories', 'origin_day', 'train_start_day', 'fit_start_day']:
            if loaded[name] != expected[name]:
                raise ValueError(f'Bundle nạp lại sai {name}.')
    for name in ['classifier', 'regressor']:
        component = getattr(loaded['model'], name)
        separate = joblib.load(model_path.parent / f'{name}.joblib')
        reference = getattr(expected['model'], name) if expected is not None else component
        if hasattr(component, 'booster_'):
            text = component.booster_.model_to_string()
            if (text != separate.booster_.model_to_string()
                    or text != reference.booster_.model_to_string()
                    or text != (model_path.parent / f'{name}.txt').read_text(encoding='utf-8')):
                raise ValueError(f'Model {name} nạp lại không khớp.')
            if (component.booster_.feature_name() != loaded['features']
                    or component.booster_.pandas_categorical != [loaded['categories']]):
                raise ValueError(f'Thứ tự biến/category của {name} không khớp bundle.')
        elif not (component.value == separate.value == reference.value):
            raise ValueError(f'Mô hình hằng {name} không khớp.')
    return loaded


def train_model(training_data, parameters, model_path):
    X, y = training_data['X'], training_data['y']
    if len(X) == 0 or len(X) != len(y) or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Nhãn Train rỗng, sai kích thước hoặc không hợp lệ.')
    positive = y > 0
    start = perf_counter()
    if positive.nunique() == 1:
        classifier = ConstantModel(float(positive.iloc[0]))
        classifier_policy = 'constant_one' if positive.iloc[0] else 'constant_zero'
    else:
        classifier = LGBMClassifier(**parameters['classifier'])
        classifier.fit(X, positive.astype('int8'), categorical_feature=['item_id'])
        classifier_policy = 'lightgbm_binary'
    if positive.any():
        regressor = LGBMRegressor(**parameters['regressor'])
        regressor.fit(X.loc[positive], y.loc[positive], categorical_feature=['item_id'])
        regressor_policy = 'lightgbm_positive_only'
    else:
        regressor = ConstantModel(0)
        regressor_policy = 'constant_zero_no_positive_rows'
    elapsed = perf_counter() - start
    bundle = {'model': TwoPartModel(classifier, regressor),
              'features': training_data['features'], 'categories': training_data['categories'],
              'origin_day': training_data['fit_end_day'], 'train_start_day': 1,
              'fit_start_day': training_data['fit_start_day']}
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    for name, component in [('classifier', classifier), ('regressor', regressor)]:
        joblib.dump(component, model_path.parent / f'{name}.joblib')
        if hasattr(component, 'booster_'):
            # Python ghi Unicode path an toàn trên Windows; native save_model có thể lỗi dấu.
            (model_path.parent / f'{name}.txt').write_text(
                component.booster_.model_to_string(), encoding='utf-8')
    verify_saved_models(model_path, expected=bundle)
    info = {
        'training_time_seconds': elapsed, 'n_train_rows': len(X),
        'n_positive_rows': int(positive.sum()), 'n_zero_rows': int((~positive).sum()),
        'classifier_policy': classifier_policy, 'regressor_policy': regressor_policy,
        'n_items': int(X.item_id.nunique()), 'features': training_data['features'],
        'fit_start_day': training_data['fit_start_day'], 'fit_end_day': training_data['fit_end_day'],
        'dropped_warmup_rows': training_data['dropped_warmup_rows'],
        'effective_model_params': {'classifier': classifier.get_params(), 'regressor': regressor.get_params()},
        'actual_trees': {name: int(component.booster_.num_trees()) if hasattr(component, 'booster_') else 0
                         for name, component in [('classifier', classifier), ('regressor', regressor)]},
        'saved_model_reload_verified': True,
    }
    LOGGER.info('Two-Part: %d dòng phân loại, %d dòng hồi quy; %.2fs', len(X), positive.sum(), elapsed)
    return bundle, info


def train_classifier_reuse_regressor(training_data, classifier_parameters, source_model_path, model_path):
    """C1: chỉ fit classifier; nạp và sao chép nguyên regressor C0."""
    import hashlib
    import shutil
    source_model_path, model_path = Path(source_model_path), Path(model_path)
    source = verify_saved_models(source_model_path)
    if (training_data['features'] != source['features'] or training_data['categories'] != source['categories']
            or training_data['fit_end_day'] != source['origin_day']):
        raise ValueError('Đặc trưng/category/origin C1 không khớp C0.')
    X, y = training_data['X'], training_data['y']
    positive = y > 0
    if len(X)==0 or not np.isfinite(y).all() or (y<0).any():
        raise ValueError('Nhãn classifier không hợp lệ.')
    start = perf_counter()
    if positive.nunique() == 1:
        classifier = ConstantModel(float(positive.iloc[0]))
    else:
        classifier = LGBMClassifier(**classifier_parameters)
        classifier.fit(X, positive.astype('int8'), categorical_feature=['item_id'])
    elapsed = perf_counter()-start
    model_path.parent.mkdir(parents=True, exist_ok=False)
    bundle = {**source, 'model': TwoPartModel(classifier, source['model'].regressor)}
    joblib.dump(bundle, model_path)
    joblib.dump(classifier, model_path.parent/'classifier.joblib')
    if hasattr(classifier,'booster_'):
        (model_path.parent/'classifier.txt').write_text(classifier.booster_.model_to_string(),encoding='utf-8')
    shutil.copy2(source_model_path.parent/'regressor.joblib',model_path.parent/'regressor.joblib')
    if (source_model_path.parent/'regressor.txt').exists():
        shutil.copy2(source_model_path.parent/'regressor.txt',model_path.parent/'regressor.txt')
    verify_saved_models(model_path,expected=bundle)
    source_hash=hashlib.sha256((source_model_path.parent/'regressor.joblib').read_bytes()).hexdigest()
    assert hashlib.sha256((model_path.parent/'regressor.joblib').read_bytes()).hexdigest()==source_hash
    info=dict(training_time_seconds=elapsed,classifier_training_time_seconds=elapsed,regressor_retrained=False,
        n_train_rows=len(X),n_positive_rows=int(positive.sum()),n_zero_rows=int((~positive).sum()),
        n_items=int(X.item_id.nunique()),features=source['features'],categories=source['categories'],
        fit_start_day=29,fit_end_day=1885,dropped_warmup_rows=training_data['dropped_warmup_rows'],
        classifier_policy='lightgbm_binary' if hasattr(classifier,'booster_') else 'constant',
        source_regressor_path=str((source_model_path.parent/'regressor.joblib').resolve()),
        source_regressor_sha256=source_hash,saved_model_reload_verified=True,
        effective_model_params=dict(classifier=classifier.get_params(),regressor=source['model'].regressor.get_params()),
        actual_trees=dict(classifier=int(classifier.booster_.num_trees()) if hasattr(classifier,'booster_') else 0,
                          regressor=int(source['model'].regressor.booster_.num_trees()) if hasattr(source['model'].regressor,'booster_') else 0))
    return bundle,info
