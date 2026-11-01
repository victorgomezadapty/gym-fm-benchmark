"""Synthetic gym dataset generator (100% synthetic, no real records).
The table structure and a few aggregate shapes (plan mix and prices, renewal gaps, visit days
per active month, hour and day of week profile, deny rates, personal training package mix)
were set by hand to resemble a typical Riyadh gym. No real row, name, phone or ID is used.
Seeded (42) and reproducible."""
import numpy as np, pandas as pd, os
rng = np.random.default_rng(42)
OUT = os.path.join(os.path.dirname(__file__), "data")
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-08-31")
DAYS = pd.date_range(START, END, freq="D"); ND = len(DAYS)
N_MEMBERS = 3000
VAT = 0.15

# plans (dimension)
plans = pd.DataFrame([
 # plan_id, name, type, months, net_price, weight
 (1, "Monthly membership 595", "monthly", 1, 595.00, 0.52),
 (2, "Monthly membership 895", "monthly", 1, 778.26, 0.08),
 (3, "12M upfront membership", "annual", 12, 10400.00, 0.20),
 (4, "12M upfront membership (promo)", "annual", 12, 9500.00, 0.07),
 (5, "6M upfront membership", "semiannual", 6, 5995.00, 0.04),
 (6, "Complimentary 10 days free", "trial", 0, 0.00, 0.06),
 (7, "Tourist 7 day package", "tourist", 0, 1034.78, 0.015),
 (8, "Tourist 3 day package", "tourist", 0, 686.96, 0.015),
], columns=["plan_id","plan_name","plan_type","duration_months","net_price","w"])
PW = plans.w.values / plans.w.sum()
PLAN = plans.set_index("plan_id")

# helpers
RAMADAN = [("2023-03-23","2023-04-20"),("2024-03-11","2024-04-09"),("2025-03-01","2025-03-29"),("2026-02-18","2026-03-19")]
season = np.ones(ND)
for a,b in RAMADAN:
    season[(DAYS>=a)&(DAYS<=b)] = 0.65
season *= 1 + 0.08*np.cos(2*np.pi*(DAYS.day_of_year.values-200)/365)   # mild summer peak
DOW = np.array([1.14,1.06,1.00,0.86,0.71,0.94,1.11])  # Mon..Sun (Python dayofweek), hand-set weekly profile
season *= DOW[DAYS.day_of_week.values]
HOURS = np.arange(24)
HP = np.array([.006,.002,0,0,0,.005,.032,.04,.033,.032,.033,.033,.04,.056,.058,.068,.084,.091,.092,.092,.082,.063,.04,.018]); HP/=HP.sum()
PAYM = ["CreditCard","CreditCard - Master Card","Online - Visa","Online - MADA","CreditCard - Visa","Online - Master Card"]
PAYW = np.array([.66,.13,.076,.05,.028,.025]); PAYW/=PAYW.sum()
SRC = ["Tablet 1","Tablet 2","Online","Bar terminal"]; SRCW=[.38,.35,.21,.06]
didx = lambda ts: int((ts-START).days)

# join dates: growth over time (more joins later), weekly noise
jw = np.linspace(1.0, 1.8, ND) * season
jw /= jw.sum()
join_idx = np.sort(rng.choice(ND-20, N_MEMBERS, p=jw[:ND-20]/jw[:ND-20].sum()))

members, contracts, payments, visits, pt_packages, pt_sessions = [],[],[],[],[],[]
trainers = pd.DataFrame({"trainer_id":[f"T{i:02d}" for i in range(1,26)],
    "role": rng.choice(["Fitness Trainer","Personal Trainer","Swimming Coach"],25,p=[.4,.5,.1]),
    "hire_date": pd.to_datetime(rng.choice(pd.date_range("2021-01-01","2025-06-30"),25))})
tw = rng.dirichlet(np.ones(25)*1.5)
PTPK = [("5 PT Sessions",5,1434.78,.19),("10 PT Sessions",10,2608.70,.34),("20 PT Sessions",20,4878.26,.36),
        ("50 PT Sessions",50,11478.26,.06),("5 Swimming PT",5,1043.48,.05)]
PTW = np.array([p[3] for p in PTPK]); PTW/=PTW.sum()
cid=pid=vid=kid=sid=0

for m in range(N_MEMBERS):
    mid = f"M{m+1:05d}"
    j = int(join_idx[m]); jdate = DAYS[j]
    age = int(np.clip(rng.normal(35.8, 8.5), 18, 70))
    home = rng.choice(["CLUB_A","CLUB_B","CLUB_C"], p=[.69,.30,.01])
    corporate = rng.random() < 0.03
    eng = rng.lognormal(np.log(1.6), 0.65)          # latent visits/week
    loyalty = rng.beta(2.2, 1.0)                    # drives renewal
    pref_h = rng.choice(HOURS, p=HP)
    members.append(dict(member_id=mid, join_date=jdate.normalize(), age=age, home_club=home,
                        gender="Male", is_corporate=corporate))
    # contract chain
    plan_id = int(rng.choice(plans.plan_id, p=PW))
    t = j; active = np.zeros(ND, bool); churn_day = None
    first = True
    while t < ND:
        P = PLAN.loc[plan_id]
        if P.plan_type == "trial": length = 10
        elif P.plan_type == "tourist": length = 7 if "7" in P.plan_name else 3
        else: length = int(round(P.duration_months*30.4))
        s = DAYS[t]; e = DAYS[min(t+length-1, ND-1)]
        cid += 1; contract_id = f"C{cid:06d}"
        contracts.append(dict(contract_id=contract_id, member_id=mid, plan_id=plan_id, start_date=s, end_date=DAYS[t+length-1] if t+length-1<ND else s+pd.Timedelta(days=length-1)))
        active[t:min(t+length, ND)] = True
        if P.net_price > 0:
            pid += 1
            disc = rng.random() < 0.01
            net = round(P.net_price*(0.85 if disc else 1), 2)
            payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=contract_id, payment_time=s+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP)), minutes=int(rng.integers(60))),
                product_type="Membership", plan_id=plan_id, net_amount=net, vat_amount=round(net*VAT,2), gross_amount=round(net*(1+VAT),2),
                payment_method=rng.choice(PAYM,p=PAYW), source=rng.choice(SRC,p=SRCW), has_discount=disc, is_refund=False))
            if first and P.plan_type=="monthly" and rng.random()<0.15:   # admin/joining fee
                pid += 1; fee = float(rng.choice([2880.0, 3745.0]))
                payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=contract_id, payment_time=s+pd.Timedelta(hours=12), product_type="AdminFee", plan_id=plan_id,
                    net_amount=fee, vat_amount=round(fee*VAT,2), gross_amount=round(fee*(1+VAT),2), payment_method=rng.choice(PAYM,p=PAYW), source=rng.choice(SRC,p=SRCW), has_discount=False, is_refund=False))
        first = False
        t_end = t + length
        if t_end >= ND: break
        # renewal decision: depends on loyalty and engagement
        if P.plan_type in ("trial","tourist"):
            p_ren = 0.35*loyalty + 0.1; next_plan = int(rng.choice([1,2,3,4],p=[.6,.1,.2,.1]))
        elif P.plan_type == "monthly":
            p_ren = np.clip(0.42 + 0.42*loyalty + 0.04*min(eng,3), 0.35, 0.95); next_plan = plan_id if rng.random()<0.95 else 3
        else:
            p_ren = np.clip(0.30 + 0.45*loyalty + 0.04*min(eng,3), 0.2, 0.9); next_plan = int(rng.choice([plan_id,1,3],p=[.7,.2,.1]))
        if rng.random() < p_ren:
            gap = int(rng.choice([0,0,0,1,2,3,5,8]))   # same-day/short-gap renewals
            if P.plan_type=="monthly" and rng.random()<0.02:  # freeze a month
                pid+=1
                payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=contract_id, payment_time=DAYS[t_end-1]+pd.Timedelta(hours=18), product_type="Freeze", plan_id=plan_id,
                    net_amount=0.0, vat_amount=0.0, gross_amount=0.0, payment_method="CreditCard", source="Tablet 1", has_discount=False, is_refund=False))
                gap += 30
            t = t_end + gap; plan_id = next_plan
        else:
            churn_day = t_end
            # some come back after a long break (win-back)
            if rng.random() < 0.45:
                t = t_end + int(rng.integers(45, 400)); plan_id = int(rng.choice([1,3],p=[.8,.2])); loyalty *= 0.8; churn_day=None
            else: break
    # visits
    # engagement trend: decays before contract ends without renewal
    p = np.zeros(ND)
    p[active] = np.clip(eng/7.0, 0, 0.95)
    if churn_day is not None:
        w = np.arange(max(j, churn_day-45), churn_day)
        p[w] *= np.linspace(1.0, 0.25, len(w))
    # month-to-month variability and occasional 'off' months (travel, injury)
    mo = (DAYS.year.values-2023)*12 + DAYS.month.values-1
    mm_mult = rng.lognormal(0, 0.55, mo.max()+1) * (rng.random(mo.max()+1) > 0.10)
    p *= season * mm_mult[mo]
    p = np.clip(p, 0, 0.97)
    vd = np.nonzero(rng.random(ND) < p)[0]
    # denied attempts after churn or occasional debit
    den = []
    if churn_day is not None and churn_day < ND-1:
        k = rng.poisson(3.0); den = list(rng.integers(churn_day, min(ND, churn_day+40), k))
    for d in vd:
        h = int(np.clip(rng.normal(pref_h, 1.8), 5, 23)) if rng.random()<0.75 else int(rng.choice(HOURS,p=HP))
        vid += 1
        res, why = ("Allowed","")
        if rng.random() < 0.012: res, why = "Denied", rng.choice(["MembershipFreezedContract","FinancialDebit"],p=[.8,.2])
        visits.append((f"V{vid:08d}", mid, DAYS[d]+pd.Timedelta(hours=h, minutes=int(rng.integers(60))), res, why))
    for d in den:
        vid += 1; visits.append((f"V{vid:08d}", mid, DAYS[d]+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP)), minutes=int(rng.integers(60))), "Denied", "MembershipNoActiveContract"))
    # PT: free intro sessions, then paid packages
    intro = False
    if rng.random() < 0.45:
        kid += 1; d = min(j + int(rng.integers(0, 10)), ND-1); intro=True
        nm, q = ("Body Measurement Session",1) if rng.random()<0.65 else ("2 PT Sessions (Free)",2)
        pkg = f"K{kid:06d}"
        pt_packages.append(dict(package_id=pkg, member_id=mid, purchase_time=DAYS[d]+pd.Timedelta(hours=17), package_name=nm, sessions=q, net_amount=0.0, gross_amount=0.0, is_paid=False))
        for i in range(q):
            sid += 1; dd = min(d+i*3, ND-1)
            pt_sessions.append(dict(session_id=f"S{sid:07d}", package_id=pkg, member_id=mid, trainer_id=rng.choice(trainers.trainer_id, p=tw),
                session_time=DAYS[dd]+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP))), training_name=nm if q==1 else "PT Session (Free)", is_completed=rng.random()<0.9, is_cancelled=rng.random()<0.01, used_net_amount=0.0))
    p_buy = 0.07 + 0.10*intro + 0.04*min(eng,4)
    if active.any() and rng.random() < p_buy:
        ad = np.nonzero(active)[0]
        d = int(ad[min(len(ad)-1, int(rng.integers(0, min(len(ad), 120))))])
        trainer = rng.choice(trainers.trainer_id, p=tw)
        nbuy = 1 + rng.poisson(1.3*loyalty)
        for b in range(nbuy):
            if d >= ND: break
            nm, q, price, _ = PTPK[int(rng.choice(len(PTPK), p=PTW))]
            kid += 1; pkg=f"K{kid:06d}"
            pt_packages.append(dict(package_id=pkg, member_id=mid, purchase_time=DAYS[d]+pd.Timedelta(hours=int(rng.choice(HOURS,p=HP))), package_name=nm, sessions=q, net_amount=price, gross_amount=round(price*(1+VAT),2), is_paid=True))
            pid += 1
            payments.append(dict(payment_id=f"P{pid:07d}", member_id=mid, contract_id=None, payment_time=DAYS[d]+pd.Timedelta(hours=19), product_type="Personal Training", plan_id=None,
                net_amount=price, vat_amount=round(price*VAT,2), gross_amount=round(price*(1+VAT),2), payment_method=rng.choice(PAYM,p=PAYW), source=rng.choice(SRC,p=SRCW), has_discount=False, is_refund=False))
            per = rng.uniform(1.0, 3.0)  # sessions/week
            dd = d
            for i in range(q):
                dd += max(1, int(rng.exponential(7/per)))
                if dd >= ND: break
                sid += 1
                canc = rng.random()<0.008
                pt_sessions.append(dict(session_id=f"S{sid:07d}", package_id=pkg, member_id=mid, trainer_id=trainer if rng.random()<0.85 else rng.choice(trainers.trainer_id, p=tw),
                    session_time=DAYS[dd]+pd.Timedelta(hours=int(np.clip(rng.normal(pref_h,1.5),6,23))), training_name="Swimming PT" if "Swim" in nm else "1 PT Session",
                    is_completed=not canc and rng.random()<0.98, is_cancelled=canc, used_net_amount=0.0 if canc else round(price/q,2)))
            d = dd + int(rng.integers(0, 20))

members = pd.DataFrame(members)
contracts = pd.DataFrame(contracts)
payments = pd.DataFrame(payments).sort_values("payment_time").reset_index(drop=True)
# refunds ~0.13% of membership payments
ri = payments[payments.product_type=="Membership"].sample(frac=0.0013, random_state=1).index
payments.loc[ri,"is_refund"] = True
visits = pd.DataFrame(visits, columns=["visit_id","member_id","swipe_time","result","deny_reason"]).sort_values("swipe_time").reset_index(drop=True)
pt_packages = pd.DataFrame(pt_packages).sort_values("purchase_time").reset_index(drop=True)
pt_sessions = pd.DataFrame(pt_sessions).sort_values("session_time").reset_index(drop=True)
contracts["end_date"] = pd.to_datetime(contracts.end_date)
os.makedirs(OUT, exist_ok=True)
for name, df in dict(members=members, plans=plans.drop(columns="w"), contracts=contracts, payments=payments, visits=visits,
                     trainers=trainers, pt_packages=pt_packages, pt_sessions=pt_sessions).items():
    df.to_csv(f"{OUT}/{name}.csv", index=False); df.to_parquet(f"{OUT}/{name}.parquet", index=False)
    print(f"{name:12s} rows={len(df):>8,d} cols={df.shape[1]}")
