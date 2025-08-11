import xgboost as xgb

def train_xgb(X, y, params, n_classes):
    if n_classes==2:
        model = xgb.XGBClassifier(**{**params, "objective":"binary:logistic"})
        model.fit(X, y)
        return model.get_booster()
    else:
        model = xgb.XGBClassifier(**params)
        model.fit(X, y)
        return model.get_booster()

def save_json(booster, path):
    booster.save_model(path)
