#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recommend using opening & closing columns only.
The fixture file MUST contain all required feature columns (same as training).
We use closing odds for EV by default; if a closing odd is NaN, we fallback to opening.

Now with --debug: writes recommend_debug.json with coverage & candidate stats.
"""
import os, json, pickle, argparse, numpy as np, pandas as pd
from itertools import combinations

REQ_FEATURES = {
    'B365H','B365D','B365A','B365CH','B365CD','B365CA',
    'B365>2.5','B365<2.5','B365C>2.5','B365C<2.5',
    'AHh','B365AHH','B365AHA','AHCh','B365CAHH','B365CAHA'
}

def _read_any(path):
    return pd.read_excel(path) if str(path).lower().endswith(('.xlsx','.xls')) else pd.read_csv(path)

def require_columns(df):
    missing = sorted(list(REQ_FEATURES - set(df.columns)))
    if missing:
        raise ValueError(f"Fixture missing required columns: {missing}")

def engineer_features(df):
    out = df.copy()
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

    out['tot_delta_over'] = out['B365C>2.5'] - out['B365>2.5']
    out['tot_delta_under']= out['B365C<2.5'] - out['B365<2.5']
    out['tot_reld_over']  = np.where(out['B365>2.5']>0, out['B365C>2.5']/out['B365>2.5'] - 1.0, np.nan)
    out['tot_reld_under'] = np.where(out['B365<2.5']>0, out['B365C<2.5']/out['B365<2.5'] - 1.0, np.nan)

    out['ah_line_change']   = out['AHCh'] - out['AHh']
    out['ah_delta_home_odds']= out['B365CAHH'] - out['B365AHH']
    out['ah_delta_away_odds']= out['B365CAHA'] - out['B365AHA']
    out['ah_reld_home_odds'] = np.where(out['B365AHH']>0, out['B365CAHH']/out['B365AHH'] - 1.0, np.nan)
    out['ah_reld_away_odds'] = np.where(out['B365AHA']>0, out['B365CAHA']/out['B365AHA'] - 1.0, np.nan)
    return out

def load_models(models_dir='models'):
    models = {}
    for k in ['moneyline.pkl','over25.pkl','ah_home_ev.pkl','ah_away_ev.pkl','meta.json']:
        p = os.path.join(models_dir, k)
        try:
            if k.endswith('.json'):
                models['meta']=json.load(open(p,'r'))
            else:
                tag = {'moneyline.pkl':'money','over25.pkl':'over','ah_home_ev.pkl':'ahH','ah_away_ev.pkl':'ahA'}[k]
                models[tag]=pickle.load(open(p,'rb'))
        except Exception:
            pass
    return models

def expected_value_binary(p, odds):
    if not np.isfinite(odds) or odds < 1.01 or not np.isfinite(p): return np.nan
    return float(p*(odds-1.0) - (1.0-p))

def implied_prob_from_ev(ev, odds):
    if not np.isfinite(odds) or odds <= 0: return np.nan
    return float((ev + 1.0) / odds)

def kelly_fraction(p, odds):
    if not np.isfinite(odds) or odds <= 1.0 or not np.isfinite(p): return 0.0
    b = odds - 1.0; q = 1.0 - p
    f = (b*p - q)/b
    return float(max(0.0, f))

def profit_stats_per_unit(p, odds):
    mu = expected_value_binary(p, odds)
    if not np.isfinite(mu): return np.nan, np.nan
    win = (odds - 1.0); lose = -1.0
    var = p*(win-mu)**2 + (1.0-p)*(lose-mu)**2
    return mu, float(np.sqrt(max(0.0, var)))

def _coverage(df):
    import pandas as pd, numpy as np
    n = len(df)
    def all_notna(cols): 
        if not set(cols).issubset(df.columns): return 0
        m = df[cols].notna().all(axis=1)
        return int(m.sum())
    cov = {
        'n_rows': int(n),
        'moneyline_close_full': all_notna(['B365CH','B365CD','B365CA']),
        'moneyline_open_full':  all_notna(['B365H','B365D','B365A']),
        'totals_close_full':    all_notna(['B365C>2.5','B365C<2.5']),
        'totals_open_full':     all_notna(['B365>2.5','B365<2.5']),
        'ah_close_full':        all_notna(['AHCh','B365CAHH','B365CAHA']),
        'ah_open_full':         all_notna(['AHh','B365AHH','B365AHA']),
    }
    return cov

def build_candidates(df, models, top_k_singles=30, top_k_parlays=60):
    rows = []
    pre_rows = 0

    # Moneyline (use closing odds for EV)
    if models.get('money') is not None:
        use = models['meta'].get('moneyline',{}).get('features',{}).get('num',[])
        X = df[use]
        proba = models['money'].predict_proba(X)
        classes = list(models['money'].named_steps['clf'].classes_)
        pH = proba[:, classes.index('H')] if 'H' in classes else np.zeros(len(df))
        pD = proba[:, classes.index('D')] if 'D' in classes else np.zeros(len(df))
        pA = proba[:, classes.index('A')] if 'A' in classes else np.zeros(len(df))
        odds_cols = [df['B365CH'], df['B365CD'], df['B365CA']]
        for i in range(len(df)):
            match = f"{df.get('HomeTeam', pd.Series(['Match']*len(df))).iloc[i]} vs {df.get('AwayTeam', pd.Series([i]*len(df))).iloc[i]}"
            for side,p in zip(['H','D','A'], [pH[i], pD[i], pA[i]]):
                o = odds_cols[['H','D','A'].index(side)].iloc[i]
                if pd.isna(o): o = [df['B365H'], df['B365D'], df['B365A']][['H','D','A'].index(side)].iloc[i]
                odds = float(o) if pd.notna(o) else np.nan
                ev = expected_value_binary(p, odds)
                rows.append({'Type':'Single','Market':'Moneyline','Pick':side,'Odds':odds,'p':float(p) if np.isfinite(p) else np.nan,'EV':ev,'Games':[match],'Desc':f"{match} — 1X2 {side}"})
    # Totals 2.5
    if models.get('over') is not None:
        use = models['meta'].get('over25',{}).get('features',{}).get('num',[])
        X = df[use]
        p_over = models['over'].predict_proba(X)[:,1]
        for i in range(len(df)):
            match = f"{df.get('HomeTeam', pd.Series(['Match']*len(df))).iloc[i]} vs {df.get('AwayTeam', pd.Series([i]*len(df))).iloc[i]}"
            oo = df['B365C>2.5'].iloc[i]; ou = df['B365C<2.5'].iloc[i]
            if pd.isna(oo): oo = df['B365>2.5'].iloc[i]
            if pd.isna(ou): ou = df['B365<2.5'].iloc[i]
            po = float(p_over[i]); pu = 1.0 - po
            evo = expected_value_binary(po, float(oo) if pd.notna(oo) else np.nan)
            evu = expected_value_binary(pu, float(ou) if pd.notna(ou) else np.nan)
            rows.append({'Type':'Single','Market':'Totals2.5','Pick':'Over2.5','Odds':float(oo) if pd.notna(oo) else np.nan,'p':po,'EV':evo,'Games':[match],'Desc':f"{match} — Over 2.5"})
            rows.append({'Type':'Single','Market':'Totals2.5','Pick':'Under2.5','Odds':float(ou) if pd.notna(ou) else np.nan,'p':pu,'EV':evu,'Games':[match],'Desc':f"{match} — Under 2.5"})
    # Asian Handicap (use closing odds when available for EV)
    for side, key in [('Home','ahH'), ('Away','ahA')]:
        model = models.get(key)
        if model is None: continue
        use = models['meta'].get('ah',{}).get('features',{}).get('num',[])
        X = df[use]
        ev_pred = model.predict(X)
        odds_series = df['B365CAHH'] if side=='Home' else df['B365CAHA']
        alt_series  = df['B365AHH']  if side=='Home' else df['B365AHA']
        for i in range(len(df)):
            match = f"{df.get('HomeTeam', pd.Series(['Match']*len(df))).iloc[i]} vs {df.get('AwayTeam', pd.Series([i]*len(df))).iloc[i]}"
            o = odds_series.iloc[i]; 
            if pd.isna(o): o = alt_series.iloc[i]
            odds = float(o) if pd.notna(o) else np.nan
            ev = float(ev_pred[i])
            p_impl = implied_prob_from_ev(ev, odds)
            line = df['AHCh'].iloc[i] if pd.notna(df['AHCh'].iloc[i]) else df['AHh'].iloc[i]
            rows.append({'Type':'Single','Market':'AsianHandicap','Pick':f"{side} {line}",'Odds':odds,'p':p_impl,'EV':ev,'Games':[match],'Desc':f"{match} — AH {side} {line}"})
    cand0 = pd.DataFrame(rows)
    pre_rows = int(len(cand0))
    cand = cand0.replace([np.inf,-np.inf],np.nan).dropna(subset=['Odds','p','EV'])
    singles = cand[cand['Type']=='Single'].sort_values('EV', ascending=False).head(top_k_singles)

    base = singles[singles['EV']>0].head(12).reset_index(drop=True)
    par_rows = []
    for i,j in combinations(range(len(base)),2):
        a = base.iloc[i]; b = base.iloc[j]
        if set(a['Games']) & set(b['Games']): continue
        pA, pB = float(a['p']), float(b['p'])
        oA, oB = float(a['Odds']), float(b['Odds'])
        p_par = pA * pB
        o_par = oA * oB
        ev_par = expected_value_binary(p_par, o_par)
        par_rows.append({'Type':'Parlay','Market':'2-leg','Pick':'N/A','Odds':o_par,'p':p_par,'EV':ev_par,'Games': list(set(a['Games'])|set(b['Games'])), 'Desc': f"{a['Desc']}  +  {b['Desc']}" })
    parlays0 = pd.DataFrame(par_rows)
    parlays = parlays0.replace([np.inf,-np.inf],np.nan).dropna(subset=['Odds','p','EV']).sort_values('EV', ascending=False).head(top_k_parlays)

    diag = {
        'candidates_before_drop': int(pre_rows),
        'candidates_after_drop': int(len(cand)),
        'dropped_for_nan_or_inf': int(pre_rows - len(cand)),
        'singles_after_drop': int(len(singles)),
        'parlays_generated_raw': int(len(parlays0)),
        'parlays_after_drop': int(len(parlays)),
        'posEV_singles': int((cand[(cand["Type"]=="Single") & (cand["EV"]>0)]).shape[0]),
        'posEV_parlays': int((parlays[parlays["EV"]>0]).shape[0]),
    }

    return pd.concat([singles, parlays], ignore_index=True), diag


def allocate_portfolio(cand, bankroll, mode='low', max_games=5, max_picks=8):
    # Caps differ by mode; scoring now Sharpe-like (mu/sd), not mu - lambda*sd.
    if mode=='low':
        cap_single = 0.06; cap_parlay = 0.02; cap_total_parlays = 0.20; kcap_single = 0.03; kcap_parlay = 0.01
    else:
        cap_single = 0.15; cap_parlay = 0.06; cap_total_parlays = 0.50; kcap_single = 0.10; kcap_parlay = 0.04

    C = cand.copy()
    mu=[]; sd=[]
    for _,r in C.iterrows():
        m,s = profit_stats_per_unit(r['p'], r['Odds'])
        mu.append(m); sd.append(s)
    C['mu']=mu; C['sd']=sd

    # Score: Sharpe-like ratio (higher is better). Keep only positive EV.
    eps = 1e-9
    C['score'] = C['mu'] / (C['sd'] + eps)
    C = C[(C['mu']>0)].sort_values('score', ascending=False).reset_index(drop=True)

    budget = bankroll
    total_parlay_stake = 0.0
    used_games=set()
    picks = []
    for _,r in C.iterrows():
        games = set(r['Games'])
        if len(used_games | games) > max_games: 
            continue
        if len(picks) >= max_picks: break
        f = kelly_fraction(r['p'], r['Odds'])
        fcap = kcap_parlay if r['Type']=='Parlay' else kcap_single
        percap = cap_parlay if r['Type']=='Parlay' else cap_single
        f = min(max(0.0, f), fcap, percap)
        stake = round(min(budget, bankroll * f), 2)
        if stake <= 0: continue
        if r['Type']=='Parlay' and (total_parlay_stake + stake) > bankroll*cap_total_parlays:
            remain = bankroll*cap_total_parlays - total_parlay_stake
            if remain <= 0: continue
            stake = round(min(stake, remain, budget), 2)
            if stake <= 0: continue

        picks.append({
            'Strategy': mode,
            'Type': r['Type'], 'Market': r['Market'], 'Description': r['Desc'],
            'Odds': float(r['Odds']), 'p': float(r['p']), 'EV_per_unit': float(r['mu']),
            'Stake': stake, 'StakePct': round(stake/bankroll,4)
        })
        budget -= stake
        if r['Type']=='Parlay': total_parlay_stake += stake
        used_games |= games
        if budget <= bankroll*0.02: break
    port = pd.DataFrame(picks)
    if port.empty:
        return port, {'bankroll':bankroll,'stake_sum':0,'exp_profit':0,'risk_std':0,'n_picks':0}
    exp_profit = float(np.sum(port['Stake'] * port['EV_per_unit']))
    sds = []
    for _,pr in port.iterrows():
        m, s = profit_stats_per_unit(pr['p'], pr['Odds'])
        sds.append(s * pr['Stake'])
    risk_std = float(np.sqrt(np.sum(np.square(sds))))
    summary = {'bankroll':bankroll,'stake_sum': float(port['Stake'].sum()), 'exp_profit': exp_profit, 'risk_std': risk_std, 'n_picks': int(len(port))}
    return port, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fixture_file', required=True)
    ap.add_argument('--models_dir', default='models')
    ap.add_argument('--bankroll', type=float, default=1000.0)
    ap.add_argument('--max_games', type=int, default=5)
    ap.add_argument('--max_picks', type=int, default=8)
    ap.add_argument('--top_k_singles', type=int, default=30)
    ap.add_argument('--top_k_parlays', type=int, default=60)
    ap.add_argument('--debug', action='store_true', help='Write recommend_debug.json with coverage & candidate stats')
    args = ap.parse_args()

    df = _read_any(args.fixture_file)
    require_columns(df)
    models = load_models(args.models_dir)
    df = engineer_features(df)

    cand, diag = build_candidates(df, models, top_k_singles=args.top_k_singles, top_k_parlays=args.top_k_parlays)

    low_port, low_sum = allocate_portfolio(cand, args.bankroll, mode='low', max_games=args.max_games, max_picks=args.max_picks)
    high_port, high_sum = allocate_portfolio(cand, args.bankroll, mode='high', max_games=args.max_games, max_picks=args.max_picks)

    low_port.to_csv('portfolio_low.csv', index=False)
    high_port.to_csv('portfolio_high.csv', index=False)
    import pandas as _pd
    _pd.DataFrame([low_sum]).to_csv('portfolio_low_summary.csv', index=False)
    _pd.DataFrame([high_sum]).to_csv('portfolio_high_summary.csv', index=False)

    if args.debug:
        cov = _coverage(df)
        debug = {
            'fixture_rows': cov['n_rows'],
            'models_loaded': {k: bool(v) for k,v in {'money':models.get('money'), 'over':models.get('over'), 'ahH':models.get('ahH'), 'ahA':models.get('ahA')}.items()},
            'coverage': cov,
            'candidates': diag,
            'summaries': {'low': low_sum, 'high': high_sum},
            'notes': 'Candidates are filtered to EV>0 and risk-adjusted score > 0 before allocation.'
        }
        with open('recommend_debug.json','w') as f:
            json.dump(debug, f, indent=2, default=float)
        print('[recommend] Debug written to recommend_debug.json')

    print("[recommend] Wrote portfolio_low.csv & portfolio_high.csv and summaries.")

if __name__ == '__main__':
    main()
