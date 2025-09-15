#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train models using opening & closing odds/lines only (exact columns).
Skips any source file that is missing ANY of the specified feature columns.

Required feature columns (must exist in each training file):
- Moneyline open/close: B365H, B365D, B365A, B365CH, B365CD, B365CA
- Totals open/close:   B365>2.5, B365<2.5, B365C>2.5, B365C<2.5
- Asian Handicap open/close: AHh, B365AHH, B365AHA, AHCh, B365CAHH, B365CAHA

Targets (labels): FTR (for moneyline), FTHG+FTAG (for totals & AH). If labels are missing, that market is skipped.
No date features; no backtests.
"""
import os, glob, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, average_precision_score, r2_score, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

REQ_FEATURES = {
    'B365H','B365D','B365A','B365CH','B365CD','B365CA',
    'B365>2.5','B365<2.5','B365C>2.5','B365C<2.5',
    'AHh','B365AHH','B365AHA','AHCh','B365CAHH','B365CAHA'
}
# Label columns (not part of skip criteria above)
LBL_MONEY = {'FTR'}
LBL_TOTALS = {'FTHG','FTAG'}
LBL_AH     = {'FTHG','FTAG','AHh'}

def _read_any(path):
    return pd.read_excel(path) if str(path).lower().endswith(('.xlsx','.xls')) else pd.read_csv(path)

def load_all_data(data_dir="./data"):
    files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.xlsx")) + glob.glob(os.path.join(data_dir, "*.xls"))
    if not files:
        print("[train_model] No files found in", data_dir)
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = _read_any(f)
            if not REQ_FEATURES.issubset(df.columns):
                missing = sorted(list(REQ_FEATURES - set(df.columns)))
                print(f"[train_model] Skipping {os.path.basename(f)} (missing feature cols: {missing})")
                continue
            df['__source__'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"[train_model] Failed to read {f}: {e}")
    if not dfs:
        print("[train_model] No usable files after filtering.")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True, sort=False)

def engineer_features(df):
    out = df.copy()
    # Moneyline deltas & aggregates
    out['ml_open_gap']  = out['B365A']  - out['B365H']
    out['ml_close_gap'] = out['B365CA'] - out['B365CH']
    out['ml_gap_change']= out['ml_close_gap'] - out['ml_open_gap']
    out['ml_open_min']  = out[['B365H','B365D','B365A']].min(axis=1)
    out['ml_open_max']  = out[['B365H','B365D','B365A']].max(axis=1)
    out['ml_close_min'] = out[['B365CH','B365CD','B365CA']].min(axis=1)
    out['ml_close_max'] = out[['B365CH','B365CD','B365CA']].max(axis=1)
    for k in ['H','D','A']:
        out[f'ml_delta_{k}'] = out[f'B365C{k}'] - out[f'B365{k}']
        out[f'ml_reld_{k}']  = np.where(out[f'B365{k}']>0, out[f'B365C{k}']/out[f'B365{k}'] - 1.0, np.nan)

    # Totals deltas
    out['tot_delta_over'] = out['B365C>2.5'] - out['B365>2.5']
    out['tot_delta_under']= out['B365C<2.5'] - out['B365<2.5']
    out['tot_reld_over']  = np.where(out['B365>2.5']>0, out['B365C>2.5']/out['B365>2.5'] - 1.0, np.nan)
    out['tot_reld_under'] = np.where(out['B365<2.5']>0, out['B365C<2.5']/out['B365<2.5'] - 1.0, np.nan)

    # Asian handicap deltas
    out['ah_line_change']   = out['AHCh'] - out['AHh']
    out['ah_delta_home_odds']= out['B365CAHH'] - out['B365AHH']
    out['ah_delta_away_odds']= out['B365CAHA'] - out['B365AHA']
    out['ah_reld_home_odds'] = np.where(out['B365AHH']>0, out['B365CAHH']/out['B365AHH'] - 1.0, np.nan)
    out['ah_reld_away_odds'] = np.where(out['B365AHA']>0, out['B365CAHA']/out['B365AHA'] - 1.0, np.nan)
    return out

def _num(cols):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler(with_mean=False))])

def moneyline_training(df):
    if LBL_MONEY - set(df.columns):
        return None, {}, {}
    y = df['FTR'].dropna()
    if y.empty:
        return None, {}, {}
    X = df.loc[y.index]
    num_cols = [
        'B365H','B365D','B365A','B365CH','B365CD','B365CA',
        'ml_open_gap','ml_close_gap','ml_gap_change','ml_open_min','ml_open_max','ml_close_min','ml_close_max',
        'ml_delta_H','ml_delta_D','ml_delta_A','ml_reld_H','ml_reld_D','ml_reld_A'
    ]
    num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return None, {}, {}
    pre = ColumnTransformer([('num', _num(num_cols), num_cols)], remainder='drop')
    pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=600, multi_class='multinomial'))])
    pipe.fit(X[num_cols], y)
    try:
        proba = pipe.predict_proba(X[num_cols])
        pred = pipe.classes_[np.argmax(proba, axis=1)]
        metrics = {'accuracy': float(accuracy_score(y, pred)), 'log_loss': float(log_loss(y, proba))}
    except Exception:
        metrics = {}
    return pipe, {'num': num_cols, 'cat': []}, metrics

def over25_training(df):
    if LBL_TOTALS - set(df.columns):
        return None, {}, {}
    y = ((df['FTHG'].fillna(-1)+df['FTAG'].fillna(-1)) > 2.5).astype(int)
    X = df.loc[y.index]
    num_cols = [
        'B365>2.5','B365<2.5','B365C>2.5','B365C<2.5',
        'tot_delta_over','tot_delta_under','tot_reld_over','tot_reld_under',
        'B365CH','B365CD','B365CA'
    ]
    num_cols = [c for c in num_cols if c in X.columns]
    if not num_cols:
        return None, {}, {}
    pre = ColumnTransformer([('num', _num(num_cols), num_cols)], remainder='drop')
    pipe = Pipeline([('pre', pre), ('clf', GradientBoostingClassifier(random_state=42))])
    pipe.fit(X[num_cols], y)
    try:
        p = pipe.predict_proba(X[num_cols])[:,1]
        metrics = {'roc_auc': float(roc_auc_score(y, p)), 'avg_precision': float(average_precision_score(y, p))}
    except Exception:
        metrics = {}
    return pipe, {'num': num_cols, 'cat': []}, metrics

def _split_quarter(line):
    if np.isfinite(line) and abs(line*2 - round(line*2)) > 1e-9:
        base = np.floor(line*2)/2.0; return [base, base+0.5]
    return [line]

def _ah_profit_for_team(goal_diff_team, line_for_team, odds):
    prof = 0.0
    parts = _split_quarter(line_for_team)
    for part in parts:
        adj = goal_diff_team + part
        if adj > 0: prof += (odds-1.0) * (0.5 if len(parts)==2 else 1.0)
        elif adj == 0: prof += 0.0
        else: prof += (-1.0) * (0.5 if len(parts)==2 else 1.0)
    return prof

def ah_training(df):
    if LBL_AH - set(df.columns):
        return None, None, {}, {}, {}
    gd_home = df['FTHG'] - df['FTAG']
    home_profit, away_profit = [], []
    for gd, line_o, line_c, oh_o, oa_o, oh_c, oa_c in zip(
        gd_home, df['AHh'], df['AHCh'], df['B365AHH'], df['B365AHA'], df['B365CAHH'], df['B365CAHA']
    ):
        if pd.isna(gd) or pd.isna(line_o):
            home_profit.append(np.nan); away_profit.append(np.nan); continue
        oh = oh_c if pd.notna(oh_c) else oh_o
        oa = oa_c if pd.notna(oa_c) else oa_o
        home_profit.append(np.nan if (pd.isna(oh) or oh<1.01) else _ah_profit_for_team(gd, line_o, oh))
        away_profit.append(np.nan if (pd.isna(oa) or oa<1.01) else _ah_profit_for_team(-gd, -line_o, oa))

    dfH = df.copy(); dfH['ah_profit']=home_profit
    dfA = df.copy(); dfA['ah_profit']=away_profit
    num_cols = [
        'AHh','AHCh','ah_line_change',
        'B365AHH','B365AHA','B365CAHH','B365CAHA',
        'ah_delta_home_odds','ah_delta_away_odds','ah_reld_home_odds','ah_reld_away_odds',
        'B365CH','B365CD','B365CA'
    ]
    num_cols = [c for c in num_cols if c in df.columns]
    if not num_cols:
        return None, None, {}, {}, {}
    pre = ColumnTransformer([('num', _num(num_cols), num_cols)], remainder='drop')
    from sklearn.pipeline import Pipeline
    mH=mA=None; metH=metA={}
    dH = dfH.dropna(subset=['ah_profit'])
    if not dH.empty:
        XH=dH[num_cols]; yH=dH['ah_profit'].values
        mH = Pipeline([('pre', pre), ('clf', GradientBoostingRegressor(random_state=42))]).fit(XH, yH)
        try:
            predH = mH.predict(XH); metH = {'r2': float(r2_score(yH, predH)), 'rmse': float(np.sqrt(mean_squared_error(yH, predH)))}
        except Exception:
            metH = {}
    dA = dfA.dropna(subset=['ah_profit'])
    if not dA.empty:
        XA=dA[num_cols]; yA=dA['ah_profit'].values
        mA = Pipeline([('pre', pre), ('clf', GradientBoostingRegressor(random_state=42))]).fit(XA, yA)
        try:
            predA = mA.predict(XA); metA = {'r2': float(r2_score(yA, predA)), 'rmse': float(np.sqrt(mean_squared_error(yA, predA)))}
        except Exception:
            metA = {}
    return mH, mA, {'num':num_cols,'cat':[]}, metH, metA

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--models_dir', default='models')
    args = ap.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    raw = load_all_data(args.data_dir)
    if raw.empty:
        print("[train_model] No usable data found."); return
    df = engineer_features(raw)

    money_pipe, money_feats, money_metrics = moneyline_training(df)
    if money_pipe: pickle.dump(money_pipe, open(os.path.join(args.models_dir,'moneyline.pkl'),'wb'))
    over_pipe, over_feats, over_metrics = over25_training(df)
    if over_pipe: pickle.dump(over_pipe, open(os.path.join(args.models_dir,'over25.pkl'),'wb'))
    ahH_pipe, ahA_pipe, ah_feats, ahH_metrics, ahA_metrics = ah_training(df)
    if ahH_pipe: pickle.dump(ahH_pipe, open(os.path.join(args.models_dir,'ah_home_ev.pkl'),'wb'))
    if ahA_pipe: pickle.dump(ahA_pipe, open(os.path.join(args.models_dir,'ah_away_ev.pkl'),'wb'))

    meta = {
        'features_required': sorted(list(REQ_FEATURES)),
        'moneyline': {'features': money_feats, 'metrics': money_metrics},
        'over25': {'features': over_feats, 'metrics': over_metrics},
        'ah': {'features': ah_feats, 'metrics_home': ahH_metrics, 'metrics_away': ahA_metrics}
    }
    json.dump(meta, open(os.path.join(args.models_dir,'meta.json'),'w'), indent=2, default=float)
    print("[train_model] Trained. Metrics:", json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
