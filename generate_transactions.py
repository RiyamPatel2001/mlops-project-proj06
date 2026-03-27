"""
CEX Transaction Generator
ActualBudget MLOps Project — Data Pipeline

Usage:
    python generate_transactions_final.py --year 2022 --input_files fmli222.csv fmli223.csv fmli224.csv fmli231.csv --output transactions_2022.csv
    python generate_transactions_final.py --year 2023 --input_files fmli232.csv fmli233.csv fmli234.csv fmli241.csv --output transactions_2023.csv
    python generate_transactions_final.py --year 2024 --input_files fmli241x.csv fmli242.csv fmli243.csv fmli244.csv fmli251.csv --output transactions_2024.csv
"""

import csv
import random
import argparse
import statistics
from datetime import datetime, timedelta
from collections import Counter

random.seed(42)

CATEGORY_MAP = {
    'Groceries':         ['FDHOMECQ'],
    'Dining Out':        ['FDAWAYCQ'],
    'Utilities':         ['ELCTRCCQ', 'NTLGASCQ', 'WATRPSCQ'],
    'Phone & Internet':  ['TELEPHCQ'],
    'Transport':         ['GASMOCQ', 'MAINRPCQ', 'VRNTLOCQ'],
    'Vehicle Payment':   ['VEHFINCQ'],
    'Vehicle Insurance': ['VEHINSCQ'],
    'Healthcare':        ['MEDSRVCQ', 'PREDRGCQ', 'MEDSUPCQ'],
    'Health Insurance':  ['HLTHINCQ'],
    'Entertainment':     ['FEEADMCQ', 'OTHENTCQ'],
    'Streaming':         ['TVRDIOCQ'],
    'Clothing':          ['ADLTAPCQ', 'CHLDAPCQ', 'TEXTILCQ'],
    'Childcare':         ['BBYDAYCQ'],
    'Pets':              ['PETTOYCQ'],
    'Alcohol':           ['ALCBEVCQ'],
    'Personal Care':     ['PERSCACQ'],
    'Education':         ['EDUCACQ'],
    'Home Improvement':  ['HOUSEQCQ', 'FURNTRCQ', 'MAJAPPCQ', 'SMLAPPCQ'],
    'Insurance':         ['PERINSCQ', 'LIFINSCQ', 'MRPINSCQ'],
    'Savings':           ['RETPENCQ'],
    'Public Transit':    ['PUBTRACQ'],
    'Travel':            ['OTHLODCQ', 'TRNTRPCQ', 'TRNOTHCQ'],
    'Charitable Giving': ['CASHCOCQ'],
    'Tobacco':           ['TOBACCCQ'],
    'Rent / Mortgage':   ['RENDWECQ', 'OWNDWECQ', 'MRTINTCQ'],
    'Household Supplies':['HOUSOPCQ', 'OTHHEXCQ'],
    'Reading':           ['READCQ'],
    'Property Tax':      ['PROPTXCQ'],
    'Other':             ['MISCCQ'],
}

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

def generate_merchant(category):
    templates = MERCHANTS.get(category, ['MISC PURCHASE'])
    t = random.choice(templates)
    if '%05d' in t: return t % random.randint(10000, 99999)
    if '%04d' in t: return t % random.randint(1000, 9999)
    if '%03d' in t: return t % random.randint(100, 999)
    if '%06d' in t: return t % random.randint(100000, 999999)
    if '%08d' in t: return t % random.randint(10000000, 99999999)
    if '%s'   in t: return t % ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=6))
    return t

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

def main():
    parser = argparse.ArgumentParser(description='Generate transactions from CEX FMLI files')
    parser.add_argument('--year',        type=int,  required=True, help='Year (e.g. 2022)')
    parser.add_argument('--input_files', nargs='+', required=True, help='FMLI CSV files')
    parser.add_argument('--output',      type=str,  required=True, help='Output CSV filename')
    parser.add_argument('--n_users',     type=int,  default=500,   help='Number of synthetic users')
    args = parser.parse_args()

    print(f"Loading {args.year} FMLI files...")
    all_rows = []
    for fp in args.input_files:
        with open(fp) as f:
            all_rows.extend(list(csv.DictReader(f)))

    active = [r for r in all_rows if float(r.get('TOTEXPCQ', 0)) > 0]
    print(f"Loaded {len(active)} active households")

    sampled = random.sample(active, min(args.n_users, len(active)))
    print(f"Sampled {len(sampled)} synthetic users")

    print("Generating transactions...")
    rows_out = []
    tx_id = 1

    for user_idx, household in enumerate(sampled):
        user_id = f"user_{user_idx+1:04d}"
        for category, cex_cols in CATEGORY_MAP.items():
            quarterly_spend = sum(float(household.get(col, 0)) for col in cex_cols)
            if quarterly_spend <= 0:
                continue
            amounts = quarterly_to_transactions(quarterly_spend, category)
            if not amounts:
                continue
            dates = assign_dates(len(amounts), args.year)
            for amount, date in zip(amounts, dates):
                if amount <= 0:
                    continue
                rows_out.append({
                    'transaction_id': f'txn_{tx_id:07d}',
                    'user_id':        user_id,
                    'payee':          generate_merchant(category),
                    'amount':         round(amount, 2),
                    'date':           date.strftime('%Y-%m-%d'),
                    'day_of_week':    date.strftime('%A'),
                    'category':       category,
                    'family_size':    household.get('FAM_SIZE', ''),
                    'has_children':   '1' if int(household.get('CHILDAGE', 0)) > 0 else '0',
                })
                tx_id += 1

    rows_out = [r for r in rows_out if float(r['amount']) > 0]
    user_counts = Counter(r['user_id'] for r in rows_out)
    valid_users = {u for u, c in user_counts.items() if c >= 10}
    rows_out = [r for r in rows_out if r['user_id'] in valid_users]

    fieldnames = ['transaction_id', 'user_id', 'payee', 'amount', 'date',
                  'day_of_week', 'category', 'family_size', 'has_children']

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    amounts = [float(r['amount']) for r in rows_out]
    cat_counts = Counter(r['category'] for r in rows_out)

    print()
    print("=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Output:             {args.output}")
    print(f"Total transactions: {len(rows_out):,}")
    print(f"Total users:        {len(valid_users)}")
    print(f"Max amount:         ${max(amounts):.2f}")
    print(f"Median amount:      ${statistics.median(amounts):.2f}")
    print()
    print("Category distribution:")
    for cat, count in cat_counts.most_common():
        print(f"  {count:6,}  {count/len(rows_out)*100:4.1f}%  {cat}")

if __name__ == '__main__':
    main()
