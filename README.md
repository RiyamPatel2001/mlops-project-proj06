# ActualBudget MLOps Project — proj06

## Team
- Saketh (Data)
- Riyam (Training)
- Jayraj (Serving)
- Puneeth (DevOps)

## Data Pipeline

### Dataset
Generated from the Consumer Expenditure Survey (CEX) Public Use Microdata (PUMD), 
US Bureau of Labor Statistics. Real household spending data across 29 categories.

### Files
- `transactions_2022.csv` — Layer 1 training data (2022 CEX)
- `transactions_2023.csv` — Production simulation seed (2023 CEX)
- `transactions_2024.csv` — Layer 2 and 3 evaluation (2024 CEX)

### Reproducing the data
1. Download CEX PUMD CSV files from https://www.bls.gov/cex/pumd_data.htm
2. Extract the FMLI files for each year
3. Run:
```bash
python generate_transactions.py --year 2022 --input_files fmli222.csv fmli223.csv fmli224.csv fmli231.csv --output transactions_2022.csv

python generate_transactions.py --year 2023 --input_files fmli232.csv fmli233.csv fmli234.csv fmli241.csv --output transactions_2023.csv

python generate_transactions.py --year 2024 --input_files fmli241x.csv fmli242.csv fmli243.csv fmli244.csv fmli251.csv --output transactions_2024.csv
```

### Input features for Layer 1
- `payee` — merchant name string
- `amount` — transaction amount in USD
- `day_of_week` — day of week string

### Output
- `category` — one of 29 spending categories
