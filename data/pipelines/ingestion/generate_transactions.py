"""
CEX Transaction Generator
ActualBudget MLOps Project — Data Pipeline

Generates synthetic transactions from Consumer Expenditure Survey (CEX) PUMD data.
Uses BOTH Interview (FMLI) and Diary (FMLD) survey data as per CEX methodology:
  - Interview: major/recurring expenditures (housing, vehicles, insurance, etc.)
  - Diary: small frequent purchases (groceries, dining, alcohol, personal care, etc.)

Since Interview and Diary survey different households, each synthetic user is built by:
  1. Sampling a real Interview household (primary — provides NEWID and demographics)
  2. Finding the closest matching Diary household by demographics (family size, income, children, urban)
  3. Using Interview data for big recurring categories
  4. Using Diary data for everyday spending categories

This ensures each user looks like a complete realistic person — not just big expenses
or just everyday spending.

Merchant names are user-consistent: same user shops at same stores (80/20 primary/secondary).
Every transaction is traceable to its source CEX household via newid and diary_newid fields.

Usage:
    python generate_transactions.py \
        --year 2022 \
        --interview_files fmli222.csv fmli223.csv fmli224.csv fmli231.csv \
        --diary_files fmld221.csv fmld222.csv fmld223.csv fmld224.csv \
        --output transactions_2022.csv
"""

import csv
import random
import argparse
import statistics
from datetime import datetime, timedelta
from collections import Counter

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)

# ── CATEGORY MAP ──────────────────────────────────────────────────────────────

# Interview survey (FMLI) — quarterly amounts
INTERVIEW_CATEGORY_MAP = {
    'Utilities':         ['ELCTRCCQ', 'NTLGASCQ', 'WATRPSCQ'],
    'Phone & Internet':  ['TELEPHCQ'],
    'Transport':         ['GASMOCQ', 'MAINRPCQ', 'VRNTLOCQ'],
    'Vehicle Payment':   ['VEHFINCQ'],
    'Vehicle Insurance': ['VEHINSCQ'],
    'Health Insurance':  ['HLTHINCQ'],
    'Healthcare':        ['MEDSRVCQ', 'PREDRGCQ', 'MEDSUPCQ'],
    'Entertainment':     ['FEEADMCQ', 'OTHENTCQ'],
    'Streaming':         ['TVRDIOCQ'],
    'Clothing':          ['ADLTAPCQ', 'CHLDAPCQ', 'TEXTILCQ'],
    'Childcare':         ['BBYDAYCQ'],
    'Education':         ['EDUCACQ'],
    'Home Improvement':  ['HOUSEQCQ', 'FURNTRCQ', 'MAJAPPCQ', 'SMLAPPCQ'],
    'Insurance':         ['PERINSCQ', 'LIFINSCQ', 'MRPINSCQ'],
    'Savings':           ['RETPENCQ'],
    'Public Transit':    ['PUBTRACQ'],
    'Travel':            ['OTHLODCQ', 'TRNTRPCQ', 'TRNOTHCQ'],
    'Charitable Giving': ['CASHCOCQ'],
    'Rent / Mortgage':   ['RENDWECQ', 'OWNDWECQ', 'MRTINTCQ'],
    'Reading':           ['READCQ'],
    'Property Tax':      ['PROPTXCQ'],
    'Other':             ['MISCCQ'],
}

# Diary survey (FMLD) — weekly amounts, multiply by 13 for quarterly equivalent
DIARY_CATEGORY_MAP = {
    'Groceries':         ['FOODHOME'],
    'Dining Out':        ['FOODAWAY'],
    'Alcohol':           ['ALCBEV'],
    'Tobacco':           ['SMOKSUPP'],
    'Pets':              ['PET_FOOD'],
    'Personal Care':     ['PERSPROD', 'PERSSERV'],
    'Household Supplies':['HOUSKEEP'],
}

DIARY_MULTIPLIER = 13  # weekly -> quarterly

# ── MERCHANT LISTS ────────────────────────────────────────────────────────────

MERCHANTS = {
    'Groceries':         ['WALMART GROCERY', 'WHOLE FOODS MKT', 'KROGER #%04d', 'TRADER JOES',
                          'SAFEWAY #%04d', 'PUBLIX #%04d', 'ALDI', 'COSTCO WHSE #%04d',
                          'TARGET GROCERY', 'HEB #%04d', 'WEGMANS #%04d', 'STOP AND SHOP'],
    'Dining Out':        ['MCDONALDS #%05d', 'STARBUCKS #%05d', 'CHIPOTLE #%04d',
                          'CHICK-FIL-A #%04d', 'SUBWAY #%05d', 'PANERA BREAD #%04d',
                          'DOORDASH*%s', 'UBER EATS*%s', 'DOMINOS #%04d',
                          'PIZZA HUT #%04d', 'OLIVE GARDEN #%04d', 'APPLEBEES #%04d',
                          'DUNKIN #%05d'],
    'Utilities':         ['CON EDISON', 'PG&E', 'DUKE ENERGY', 'NATIONAL GRID',
                          'AMEREN ELECTRIC', 'XCEL ENERGY', 'DOMINION ENERGY',
                          'AMERICAN WATER', 'CITY WATER DEPT', 'SOUTHWEST GAS'],
    'Phone & Internet':  ['AT&T*BILL', 'VERIZON*WIRELESS', 'T-MOBILE*AUTO PAY',
                          'COMCAST XFINITY', 'SPECTRUM*%08d', 'COX COMMUNICATIONS',
                          'GOOGLE FI', 'MINT MOBILE'],
    'Transport':         ['SHELL OIL %08d', 'EXXON MOBIL %08d', 'BP #%06d',
                          'CHEVRON #%06d', 'SUNOCO #%06d', 'CIRCLE K #%05d',
                          'SPEEDWAY #%04d', 'VALVOLINE #%04d'],
    'Vehicle Payment':   ['TOYOTA FINANCIAL', 'FORD MOTOR CREDIT', 'GM FINANCIAL',
                          'HONDA FINANCIAL', 'NISSAN MOTOR ACC', 'CARMAX AUTO FIN',
                          'ALLY FINANCIAL', 'CHASE AUTO FINANCE'],
    'Vehicle Insurance': ['STATE FARM INS', 'GEICO INSURANCE', 'PROGRESSIVE INS',
                          'ALLSTATE INS', 'USAA INSURANCE', 'FARMERS INS'],
    'Healthcare':        ['CVS PHARMACY #%05d', 'WALGREENS #%05d', 'RITE AID #%04d',
                          'LABCORP', 'QUEST DIAGNOSTICS', 'URGENT CARE CTR',
                          'KAISER PERMANENTE', 'PLANNED PARENTHOOD'],
    'Health Insurance':  ['BLUE CROSS BLUE SHIELD', 'AETNA HEALTH', 'CIGNA HEALTH',
                          'HUMANA INSURANCE', 'UNITED HEALTHCARE', 'KAISER HEALTH'],
    'Entertainment':     ['AMC THEATRES #%04d', 'REGAL CINEMAS #%04d', 'BOWLERO #%03d',
                          'DAVE AND BUSTERS', 'TOPGOLF #%03d', 'LIVE NATION',
                          'TICKETMASTER', 'STUBHUB'],
    'Streaming':         ['NETFLIX.COM', 'SPOTIFY USA', 'HULU', 'DISNEY PLUS',
                          'HBO MAX', 'APPLE.COM/BILL', 'AMAZON PRIME',
                          'PEACOCK TV', 'PARAMOUNT PLUS', 'YOUTUBE PREMIUM'],
    'Clothing':          ['AMAZON.COM*%s', 'TARGET #%04d', 'WALMART #%04d',
                          'KOHLS #%04d', 'MACYS #%04d', 'TJMAXX #%04d',
                          'ROSS STORES #%04d', 'OLD NAVY #%04d', 'GAP #%04d',
                          'NIKE.COM', 'H&M #%04d', 'ZARA #%04d'],
    'Childcare':         ['BRIGHT HORIZONS #%03d', 'KINDERCARE #%03d',
                          'LA PETITE #%03d', 'LITTLE GYM #%03d',
                          'PRIMROSE SCHOOL', 'CHILD TIME #%03d',
                          'ABC LEARNING CTR', 'SMART KIDS ACAD'],
    'Pets':              ['PETSMART #%04d', 'PETCO #%04d', 'CHEWY.COM',
                          'BANFIELD PET HOSP', 'VCA ANIMAL HOSP', 'PETLAND #%03d'],
    'Alcohol':           ['TOTAL WINE #%03d', 'BEV MO #%03d', 'SPECS #%03d',
                          'DRIZLY*ORDER', 'ABC FINE WINE #%03d', 'BINNY BEVERAGES'],
    'Personal Care':     ['GREAT CLIPS #%04d', 'SUPERCUTS #%04d', 'SPORT CLIPS #%04d',
                          'ULTA BEAUTY #%04d', 'SEPHORA #%04d', 'SALLY BEAUTY #%04d'],
    'Education':         ['UDEMY*COURSE', 'COURSERA*SUBS', 'CHEGG INC',
                          'BARNES NOBLE EDU', 'AMAZON*TEXTBOOK', 'PEARSON EDUC',
                          'COLLEGE BOARD', 'KHAN ACADEMY'],
    'Home Improvement':  ['HOME DEPOT #%04d', 'LOWES #%04d', 'MENARDS #%04d',
                          'ACE HARDWARE #%04d', 'TRUE VALUE #%04d', 'WAYFAIR*ORDER',
                          'IKEA #%03d', 'WILLIAMS SONOMA'],
    'Insurance':         ['ALLSTATE LIFE', 'STATE FARM LIFE', 'METLIFE INS',
                          'PRUDENTIAL INS', 'NEW YORK LIFE', 'TRANSAMERICA'],
    'Savings':           ['FIDELITY INVEST', 'VANGUARD', 'CHARLES SCHWAB',
                          'BETTERMENT', 'WEALTHFRONT', 'ROBINHOOD'],
    'Public Transit':    ['MTA*METROCARD', 'CTA VENTRA', 'WMATA SMARTRIP',
                          'BART TICKETS', 'MBTA CHARLIE', 'SEPTA TRANSIT',
                          'UBER*TRIP', 'LYFT*RIDE'],
    'Travel':            ['MARRIOTT #%05d', 'HILTON #%05d', 'AIRBNB*%s',
                          'AMERICAN AIRLINES', 'DELTA AIR LINES', 'UNITED AIRLINES',
                          'SOUTHWEST AIR', 'EXPEDIA*HOTEL', 'BOOKING.COM'],
    'Charitable Giving': ['RED CROSS', 'UNITED WAY', 'SALVATION ARMY',
                          'ST JUDE CHILDRENS', 'DOCTORS W/O BRDRS',
                          'LOCAL FOOD BANK', 'HABITAT HUMANITY', 'GOODWILL DONAT'],
    'Tobacco':           ['CIRCLE K TOBACCO', '7-ELEVEN #%05d', 'SPEEDWAY TOBACCO',
                          'CVS TOBACCO', 'ALTRIA GROUP', 'JUUL LABS'],
    'Rent / Mortgage':   ['RENT PAYMENT', 'MORTGAGE PAYMENT', 'WELLS FARGO MTG',
                          'CHASE MORTGAGE', 'BANK OF AMER MTG', 'QUICKEN LOANS',
                          'PROPERTY MGMT CO', 'LANDLORD PAYMENT'],
    'Household Supplies':['AMAZON.COM*%s', 'TARGET #%04d', 'WALMART #%04d',
                          'DOLLAR GENERAL #%05d', 'DOLLAR TREE #%05d',
                          'BED BATH BEYOND', 'CONTAINER STORE'],
    'Reading':           ['AMAZON KINDLE', 'BARNES NOBLE #%04d', 'AUDIBLE*CHGS',
                          'NEW YORK TIMES', 'WASHINGTON POST', 'MEDIUM.COM'],
    'Property Tax':      ['COUNTY TAX PYMNT', 'PROPERTY TAX PMT', 'CITY TAX OFFICE'],
    'Other':             ['PAYPAL*%s', 'VENMO*PAYMENT', 'CASH APP*%s',
                          'GOOGLE*SERVICES', 'APPLE.COM/BILL', 'AMAZON*MISC'],
}

TX_PROFILES = {
    'Groceries':         (40,  120),
    'Dining Out':        (12,   45),
    'Utilities':         (60,  250),
    'Phone & Internet':  (50,  150),
    'Transport':         (35,   75),
    'Vehicle Payment':   (200, 600),
    'Vehicle Insurance': (100, 400),
    'Healthcare':        (20,  200),
    'Health Insurance':  (200, 800),
    'Entertainment':     (15,   80),
    'Streaming':         (8,    20),
    'Clothing':          (25,  120),
    'Childcare':         (200, 900),
    'Pets':              (25,   90),
    'Alcohol':           (18,   70),
    'Personal Care':     (15,   55),
    'Education':         (50,  500),
    'Home Improvement':  (40,  400),
    'Insurance':         (40,  250),
    'Savings':           (100, 500),
    'Public Transit':    (3,    25),
    'Travel':            (80,  700),
    'Charitable Giving': (15,  150),
    'Tobacco':           (8,    30),
    'Rent / Mortgage':   (500, 3500),
    'Household Supplies':(15,   70),
    'Reading':           (5,    25),
    'Property Tax':      (300, 1800),
    'Other':             (8,    90),
}

AMOUNT_CAPS = {
    'Dining Out':        500,
    'Public Transit':    300,
    'Transport':         500,
    'Charitable Giving': 500,
    'Household Supplies':300,
    'Other':             300,
    'Vehicle Payment':   2000,
    'Insurance':         1500,
    'Health Insurance':  1500,
    'Rent / Mortgage':   4000,
    'Property Tax':      2000,
    'Entertainment':     400,
    'Travel':            1500,
    'Education':         2000,
    'Childcare':         2000,
}

# ── DEMOGRAPHIC MATCHING ──────────────────────────────────────────────────────

def build_diary_index(diary_rows):
    """
    Build a lookup structure for fast demographic matching.
    Groups diary households by (has_children, income_quintile, urban).
    """
    index = {}
    for r in diary_rows:
        has_children = 1 if int(r.get('CHILDAGE', 0)) > 0 else 0
        inc_rank = float(r.get('INC_RANK', 0.5))
        income_quintile = min(4, int(inc_rank * 5))  # 0-4
        urban = r.get('BLS_URBN', '1')
        key = (has_children, income_quintile, urban)
        if key not in index:
            index[key] = []
        index[key].append(r)
    return index


def find_matching_diary_household(interview_hh, diary_index, rng):
    """
    Find the closest matching diary household for an interview household.
    Matches on: has_children, income_quintile, urban status.
    Falls back to relaxed matching if no exact match found.
    """
    has_children = 1 if int(interview_hh.get('CHILDAGE', 0)) > 0 else 0
    inc_rank = float(interview_hh.get('INC_RANK', 0.5))
    income_quintile = min(4, int(inc_rank * 5))
    urban = interview_hh.get('BLS_URBN', '1')

    # Try exact match first
    key = (has_children, income_quintile, urban)
    if key in diary_index and diary_index[key]:
        return rng.choice(diary_index[key])

    # Relax income by ±1 quintile
    for q_offset in [1, -1, 2, -2]:
        relaxed_key = (has_children, max(0, min(4, income_quintile + q_offset)), urban)
        if relaxed_key in diary_index and diary_index[relaxed_key]:
            return rng.choice(diary_index[relaxed_key])

    # Relax urban constraint
    for q in range(5):
        for u in ['1', '2']:
            fallback_key = (has_children, q, u)
            if fallback_key in diary_index and diary_index[fallback_key]:
                return rng.choice(diary_index[fallback_key])

    return None


# ── USER MERCHANT PROFILE ─────────────────────────────────────────────────────

def build_user_merchant_profile(newid, categories):
    """
    Assigns each user a primary (80%) and secondary (20%) merchant per category.
    Seeded by CEX household NEWID — reproducible and traceable.
    """
    seed = int(str(newid).strip()) if str(newid).strip().isdigit() else hash(str(newid)) % (2**31)
    rng = random.Random(seed)
    profile = {}
    for category in categories:
        pool = MERCHANTS.get(category, ['MISC PURCHASE'])
        if len(pool) >= 2:
            chosen = rng.sample(pool, 2)
            profile[category] = {'primary': chosen[0], 'secondary': chosen[1]}
        else:
            profile[category] = {'primary': pool[0], 'secondary': pool[0]}
    return profile


def format_merchant(template):
    if '%05d' in template: return template % random.randint(10000, 99999)
    if '%04d' in template: return template % random.randint(1000, 9999)
    if '%03d' in template: return template % random.randint(100, 999)
    if '%06d' in template: return template % random.randint(100000, 999999)
    if '%08d' in template: return template % random.randint(10000000, 99999999)
    if '%s'   in template: return template % ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
    return template


def generate_merchant_for_user(category, user_profile):
    template = (
        user_profile[category]['primary']
        if random.random() < 0.80
        else user_profile[category]['secondary']
    )
    return format_merchant(template)


# ── TRANSACTION GENERATION ────────────────────────────────────────────────────

def quarterly_to_transactions(quarterly_spend, category):
    if quarterly_spend <= 0:
        return []
    min_tx, max_tx = TX_PROFILES.get(category, (20, 100))
    avg_tx = (min_tx + max_tx) / 2
    n = max(1, round(quarterly_spend / avg_tx))
    n = min(n, 90)
    cap = AMOUNT_CAPS.get(category, 500)
    txns = []
    remaining = quarterly_spend
    for i in range(n):
        if i == n - 1:
            amount = round(max(0.01, remaining), 2)
        else:
            amount = round(random.uniform(min_tx * 0.6, max_tx * 1.4), 2)
            amount = min(amount, remaining * 0.9)
        amount = min(amount, cap)
        if amount <= 0:
            break
        remaining -= amount
        txns.append(amount)
    return txns


def assign_dates(n, year):
    start = datetime(year, 1, 1)
    end   = datetime(year, 12, 31)
    total_days = (end - start).days
    dates = []
    for _ in range(n):
        d = start + timedelta(days=random.randint(0, total_days))
        if d.weekday() == 6 and random.random() < 0.3:
            d += timedelta(days=1)
        dates.append(d)
    return sorted(dates)


# ── DATA QUALITY CHECKS ───────────────────────────────────────────────────────

def evaluate_and_clean(rows):
    issues = []
    cleaned = []
    for r in rows:
        if float(r['amount']) <= 0:
            issues.append(f"Dropped zero/negative: {r['transaction_id']}")
            continue
        if not r['payee'] or not r['category'] or not r['date']:
            issues.append(f"Dropped missing fields: {r['transaction_id']}")
            continue
        if float(r['amount']) > 5000:
            issues.append(f"Dropped extreme amount ${r['amount']}: {r['transaction_id']}")
            continue
        cleaned.append(r)

    cat_counts = Counter(r['category'] for r in cleaned)
    total = len(cleaned)
    for cat, count in cat_counts.items():
        pct = count / total * 100
        if pct > 25:
            issues.append(f"WARNING: {cat} is {pct:.1f}% — severe class imbalance")
        if count < 10:
            issues.append(f"WARNING: {cat} has only {count} rows")

    user_counts = Counter(r['user_id'] for r in cleaned)
    low_users = {u for u, c in user_counts.items() if c < 10}
    if low_users:
        issues.append(f"Dropped {len(low_users)} users with < 10 transactions")
        cleaned = [r for r in cleaned if r['user_id'] not in low_users]

    return cleaned, issues


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate transactions from CEX FMLI + FMLD files')
    parser.add_argument('--year',            type=int,  required=True)
    parser.add_argument('--interview_files', nargs='+', required=True)
    parser.add_argument('--diary_files',     nargs='+', required=True)
    parser.add_argument('--output',          type=str,  required=True)
    parser.add_argument('--n_users',         type=int,  default=500)
    args = parser.parse_args()

    # Load interview data
    print(f"Loading {args.year} interview (FMLI) files...")
    interview_rows = []
    for fp in args.interview_files:
        with open(fp) as f:
            interview_rows.extend(list(csv.DictReader(f)))
    interview_active = [r for r in interview_rows if float(r.get('TOTEXPCQ', 0)) > 0]
    print(f"  {len(interview_active)} active interview households")

    # Load diary data
    print(f"Loading {args.year} diary (FMLD) files...")
    diary_rows = []
    for fp in args.diary_files:
        with open(fp) as f:
            diary_rows.extend(list(csv.DictReader(f)))
    diary_active = [r for r in diary_rows if float(r.get('FOODTOT', 0)) > 0]
    print(f"  {len(diary_active)} active diary households")

    # Build diary demographic index for matching
    diary_index = build_diary_index(diary_active)
    print(f"  Built diary index with {len(diary_index)} demographic groups")

    # Sample interview households
    sampled = random.sample(interview_active, min(args.n_users, len(interview_active)))
    print(f"Sampled {len(sampled)} synthetic users from interview survey")

    # Match each to a diary household
    match_rng = random.Random(GLOBAL_SEED + 1)
    diary_matches = []
    for hh in sampled:
        match = find_matching_diary_household(hh, diary_index, match_rng)
        diary_matches.append(match)
    matched = sum(1 for m in diary_matches if m is not None)
    print(f"  {matched}/{len(sampled)} users matched to diary household ({matched/len(sampled)*100:.0f}%)")

    print("Generating transactions...")
    rows_out = []
    tx_id = 1
    all_categories = list(INTERVIEW_CATEGORY_MAP.keys()) + list(DIARY_CATEGORY_MAP.keys())

    for user_idx, (interview_hh, diary_hh) in enumerate(zip(sampled, diary_matches)):
        user_id  = f"user_{user_idx+1:04d}"
        newid    = interview_hh.get('NEWID', str(user_idx))
        diary_newid = diary_hh.get('NEWID', '') if diary_hh else ''

        user_profile = build_user_merchant_profile(newid, all_categories)

        # Interview categories
        for category, cex_cols in INTERVIEW_CATEGORY_MAP.items():
            quarterly_spend = sum(float(interview_hh.get(col, 0)) for col in cex_cols)
            if quarterly_spend <= 0:
                continue
            amounts = quarterly_to_transactions(quarterly_spend, category)
            dates = assign_dates(len(amounts), args.year)
            for amount, date in zip(amounts, dates):
                if amount <= 0:
                    continue
                rows_out.append({
                    'transaction_id': f'txn_{tx_id:07d}',
                    'user_id':        user_id,
                    'newid':          newid,
                    'diary_newid':    diary_newid,
                    'survey_source':  'interview',
                    'payee':          generate_merchant_for_user(category, user_profile),
                    'amount':         round(amount, 2),
                    'date':           date.strftime('%Y-%m-%d'),
                    'day_of_week':    date.strftime('%A'),
                    'category':       category,
                    'family_size':    interview_hh.get('FAM_SIZE', ''),
                    'has_children':   '1' if int(interview_hh.get('CHILDAGE', 0)) > 0 else '0',
                })
                tx_id += 1

        # Diary categories
        if diary_hh:
            for category, diary_cols in DIARY_CATEGORY_MAP.items():
                weekly_spend = sum(float(diary_hh.get(col, 0)) for col in diary_cols)
                quarterly_spend = weekly_spend * DIARY_MULTIPLIER
                if quarterly_spend <= 0:
                    continue
                amounts = quarterly_to_transactions(quarterly_spend, category)
                dates = assign_dates(len(amounts), args.year)
                for amount, date in zip(amounts, dates):
                    if amount <= 0:
                        continue
                    rows_out.append({
                        'transaction_id': f'txn_{tx_id:07d}',
                        'user_id':        user_id,
                        'newid':          newid,
                        'diary_newid':    diary_newid,
                        'survey_source':  'diary',
                        'payee':          generate_merchant_for_user(category, user_profile),
                        'amount':         round(amount, 2),
                        'date':           date.strftime('%Y-%m-%d'),
                        'day_of_week':    date.strftime('%A'),
                        'category':       category,
                        'family_size':    diary_hh.get('FAM_SIZE', interview_hh.get('FAM_SIZE', '')),
                        'has_children':   '1' if int(diary_hh.get('CHILDAGE', 0)) > 0 else '0',
                    })
                    tx_id += 1

    # Evaluate and clean
    print("Evaluating and cleaning synthetic data...")
    rows_out, issues = evaluate_and_clean(rows_out)

    # Summary of issues (suppress individual row drops, show warnings only)
    warnings = [i for i in issues if 'WARNING' in i or 'Dropped' in i and 'users' in i]
    dropped_rows = sum(1 for i in issues if 'Dropped zero' in i or 'Dropped extreme' in i or 'Dropped missing' in i)
    if dropped_rows:
        print(f"  Dropped {dropped_rows} individual rows (zero amounts, missing fields, extremes)")
    for w in warnings:
        print(f"  {w}")

    fieldnames = ['transaction_id', 'user_id', 'newid', 'diary_newid', 'survey_source',
                  'payee', 'amount', 'date', 'day_of_week', 'category',
                  'family_size', 'has_children']

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    amounts = [float(r['amount']) for r in rows_out]
    cat_counts = Counter(r['category'] for r in rows_out)
    user_counts = Counter(r['user_id'] for r in rows_out)
    source_counts = Counter(r['survey_source'] for r in rows_out)

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Output:             {args.output}")
    print(f"Total transactions: {len(rows_out):,}")
    print(f"Total users:        {len(user_counts)}")
    print(f"Interview txns:     {source_counts.get('interview', 0):,}")
    print(f"Diary txns:         {source_counts.get('diary', 0):,}")
    print(f"Max amount:         ${max(amounts):.2f}")
    print(f"Median amount:      ${statistics.median(amounts):.2f}")
    print()
    print("Category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {count:6,}  {count/len(rows_out)*100:4.1f}%  {cat}")


if __name__ == '__main__':
    main()
