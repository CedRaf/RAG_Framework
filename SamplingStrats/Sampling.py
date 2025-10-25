import yfinance as yf
import pandas as pd
import random
import os
import shutil
import json
import glob
from collections import defaultdict



tickers_by_sector = {
    "Industrials": [
        'CTAS', 'PAYX', 'ODFL', 'ADP', 'CSX', 'HON', 'AXON', 'VRSK', 'TSLA', 'PYPL'
    ],
    "Telecommunications": [
        'CMCSA', 'CHTR', 'CSCO', 'TMUS'
    ],
    "Utilities": [
        'AEP', 'XEL', 'EXC', 'CEG'
    ],
    "BasicMaterials": [
        'LIN'
    ],
    "ConsumerStaples": [
        'MNST', 'PEP', 'MDLZ', 'KDP', 'KHC', 'CCEP'
    ],
    "Healthcare": [
        'AMGN', 'REGN', 'IDXX', 'VRTX', 'BIIB', 'GILD', 'ISRG', 'AZN', 'DXCM', 'GEHC'
    ],
    "ConsumerDiscretionary": [
        'PCAR', 'ROST', 'FAST', 'EA', 'SBUX', 'ORLY', 'COST', 'CPRT', 'TTWO', 'AMZN', 
        'BKNG', 'MAR', 'NFLX', 'LULU', 'BKR', 'ABNB', 'WBD', 'TRI', 'MELI'
    ],
    "Technology": [
        'AAPL', 'AMAT', 'INTC', 'KLAC', 'LRCX', 'ADSK', 'MSFT', 'ADBE', 'QCOM', 'SNPS',
        'INTU', 'MCHP', 'ASML', 'MSTR', 'CTSH', 'NVDA', 'ON', 'MRVL', 'ADI', 'AMD', 
        'CDNS', 'MU', 'ROP', 'TXN', 'GOOGL', 'AVGO', 'FTNT', 'NXPI', 'META', 'PANW',
        'WDAY', 'CDW', 'GOOG', 'SHOP', 'TEAM', 'TTD', 'ZS', 'PDD', 'CRWD', 'DDOG',
        'PLTR', 'DASH', 'APP', 'GFS', 'ARM'
    ],
    "RealEstate": [
        'CSGP'
    ],
    "Energy": [
        'FANG'
    ]
}

all_tickers = [ticker for tickers in tickers_by_sector.values() for ticker in tickers]
ticker_to_sector = {ticker: sector for sector, tickers in tickers_by_sector.items() for ticker in tickers}

def fetch_and_categorize_nasdaq100():
    data = []
    for ticker in all_tickers:
        stock = yf.Ticker(ticker)
        info = stock.info
        data.append({
            'Ticker': ticker,
            'Company': info.get('longName', ''),
            'Sector': ticker_to_sector.get(ticker, info.get('sector', '')),
            'Market_Cap': info.get('marketCap', 0)
        })

    df = pd.DataFrame(data)
    # Categorize market cap
    df['Market_Cap_Category'] = df['Market_Cap'].apply(
        lambda x: 'Large' if x > 200_000_000_000 else ('Mid' if x >= 50_000_000_000 else 'Small')
    )
    # Sort by sector and market cap
    df = df.sort_values(by=['Sector', 'Market_Cap'], ascending=[True, False])
    df.to_csv('nasdaq_100_companies.csv', index=False)

def stratified_sample_companies():
    df = pd.read_csv('nasdaq_100_companies.csv')
    
    # Define sampling distribution
    sampling_plan = {
        'Technology': {'Large': 6, 'Mid': 6, 'Small': 4},
        'ConsumerDiscretionary': {'Large': 3, 'Mid': 3, 'Small': 1},
        'Healthcare': {'Large': 0, 'Mid': 2, 'Small': 1},
        'Industrials': {'Large': 0, 'Mid': 1, 'Small': 2},
        'ConsumerStaples': {'Large': 1, 'Mid': 1, 'Small': 0},
        'Telecommunications': {'Large': 1, 'Mid': 0, 'Small': 0},
        'Utilities': {'Large': 0, 'Mid': 1, 'Small': 0},
        'BasicMaterials': {'Large': 1, 'Mid': 0, 'Small': 0},
        'RealEstate': {'Large': 0, 'Mid': 0, 'Small': 1}
    }
    
    sampled_companies = []
    
    for sector, caps in sampling_plan.items():
        for cap_category, count in caps.items():
            if count > 0:
                # Filter by sector and market cap category
                candidates = df[(df['Sector'] == sector) & (df['Market_Cap_Category'] == cap_category)]
                
                # Randomly sample the specified count
                if len(candidates) >= count:
                    sample = candidates.sample(n=count, random_state=None)  # random_state=None for true randomness
                else:
                    # If not enough candidates, take all available
                    print(f"Warning: Only {len(candidates)} companies in {sector}/{cap_category}, need {count}")
                    sample = candidates
                
                sampled_companies.append(sample)
    
    # Combine all samples
    result = pd.concat(sampled_companies, ignore_index=True)
    result = result.sort_values(by=['Sector', 'Market_Cap'], ascending=[True, False])
    
    # Save to file
    result.to_csv('sampled_companies.csv', index=False)
    
    print(f"\nSampled {len(result)} companies:")
    print(result.groupby(['Sector', 'Market_Cap_Category']).size())
    
    return result

def sample_edgar_filings():

    
    # Load ticker to CIK mapping
    mapping_file = "../company_tickers_exchange.json"
    if not os.path.exists(mapping_file):
        print(f"Error: {mapping_file} not found")
        return []
    
    with open(mapping_file, 'r') as f:
        ticker_data = json.load(f)
    
    # Create ticker to CIK mapping
    # The JSON structure is: {"data": [[cik, company_name, ticker, exchange], ...]}
    ticker_to_cik = {}
    
    if isinstance(ticker_data, dict) and 'data' in ticker_data:
        for entry in ticker_data['data']:
            if isinstance(entry, list) and len(entry) >= 3:
                cik = str(entry[0])  # CIK is first element, no padding
                ticker = entry[2]  # Ticker is third element
                if ticker and cik:
                    ticker_to_cik[ticker] = cik
    
    print(f"Loaded {len(ticker_to_cik)} ticker-to-CIK mappings")
    
    # Target tickers from your sampled companies
    # NOTE: this varies depending on which 35 companies were randomly chosen
    target_tickers = [
        "LIN","AMZN","NFLX","COST","MELI","SBUX","ORLY","BKR","PEP","MDLZ",
        "GILD","VRTX","BIIB","CSX","VRSK","ODFL","CSGP","AAPL","GOOGL","META",
        "AVGO","CRWD","APP","NVDA","INTU","ADBE","MSTR","MRVL","NXPI","ZS",
        "MCHP","CTSH","ON","TMUS","CEG"
    ]
    
    # Get CIKs for target tickers
    target_ciks = []
    for ticker in target_tickers:
        if ticker in ticker_to_cik:
            target_ciks.append(ticker_to_cik[ticker])
        else:
            print(f"Warning: CIK not found for ticker {ticker}")
    
    print(f"Found CIKs for {len(target_ciks)} out of {len(target_tickers)} tickers")
    
    # Debug: Show first few CIKs
    print(f"\nSample CIKs being searched for:")
    for i, (ticker, cik) in enumerate(list(zip(target_tickers, target_ciks))[:5]):
        print(f"  {ticker}: {cik}")
    
    # Define sampling distribution
    distribution = {
        '10-K': {2020: 12, 2021: 9, 2022: 12, 2023: 13, 2024: 14},
        '10-Q': {2020: 16, 2021: 12, 2022: 16, 2023: 18, 2024: 18},
        '8-K': {2020: 12, 2021: 9, 2022: 12, 2023: 13, 2024: 14}
    }
    
    # Path to edgar filings
    edgar_base_path = "../edgar-crawler/datasets/RAW_FILINGS"
    
    selected_files = []
    selection_log = []
    
    for form_type, year_distribution in distribution.items():
        form_path = os.path.join(edgar_base_path, form_type)
        
        if not os.path.exists(form_path):
            print(f"Warning: Directory not found: {form_path}")
            continue
        
        # Get all files for this form type
        all_files = glob.glob(os.path.join(form_path, "*.htm"))
        
        print(f"\nTotal {form_type} files in directory: {len(all_files)}")
        
        # Debug: Show first few filenames
        if all_files:
            print(f"Sample filenames:")
            for file_path in all_files[:3]:
                filename = os.path.basename(file_path)
                print(f"  {filename}")
        
        # Filter files by target CIKs and organize by year
        files_by_year = defaultdict(list)
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Parse filename: {CIK}_{FormType}_{Year}_{AccessionNumber}.htm
            parts = filename.split('_')
            if len(parts) >= 3:
                file_cik = parts[0]
                file_year = int(parts[2]) if parts[2].isdigit() else None
                
                # Debug: Show first file's CIK comparison
                if file_path == all_files[0]:
                    print(f"\nFirst file CIK: '{file_cik}' (length: {len(file_cik)})")
                    print(f"Is in target CIKs: {file_cik in target_ciks}")
                    if target_ciks:
                        print(f"Sample target CIK: '{target_ciks[0]}' (length: {len(target_ciks[0])})")
                
                # Check if this file is from one of our target companies
                if file_cik in target_ciks and file_year in year_distribution:
                    files_by_year[file_year].append(file_path)
        
        print(f"\n{form_type} files found by year:")
        for year in sorted(files_by_year.keys()):
            print(f"  {year}: {len(files_by_year[year])} files")
        
        # Sample files according to distribution
        for year, target_count in year_distribution.items():
            available_files = files_by_year.get(year, [])
            
            if len(available_files) >= target_count:
                sampled = random.sample(available_files, target_count)
            else:
                print(f"Warning: Only {len(available_files)} {form_type} files available for {year}, need {target_count}")
                sampled = available_files
            
            selected_files.extend(sampled)
            
            # Log selection details
            for file_path in sampled:
                filename = os.path.basename(file_path)
                parts = filename.split('_')
                cik = parts[0] if len(parts) > 0 else 'unknown'
                
                # Find ticker for this CIK
                ticker = 'unknown'
                for t, c in ticker_to_cik.items():
                    if c == cik:
                        ticker = t
                        break
                
                selection_log.append({
                    'form_type': form_type,
                    'year': year,
                    'ticker': ticker,
                    'cik': cik,
                    'filename': filename,
                    'full_path': file_path
                })
    
    # Create summary
    print(f"\n=== SAMPLING SUMMARY ===")
    print(f"Total files selected: {len(selected_files)}")
    
    # Summary by form type and year
    summary = defaultdict(lambda: defaultdict(int))
    for log_entry in selection_log:
        summary[log_entry['form_type']][log_entry['year']] += 1
    
    for form_type in ['10-K', '10-Q', '8-K']:
        total = sum(summary[form_type].values())
        print(f"\n{form_type}: {total} files")
        for year in sorted(summary[form_type].keys()):
            count = summary[form_type][year]
            target = distribution[form_type].get(year, 0)
            print(f"  {year}: {count}/{target} files")
    
    # Save selection log to CSV
    log_df = pd.DataFrame(selection_log)
    log_df.to_csv('edgar_document_selection.csv', index=False)
    print(f"\nSelection log saved to: edgar_document_selection.csv")
    
    # Save just the file paths for easy processing
    with open('selected_edgar_files.txt', 'w') as f:
        for file_path in selected_files:
            f.write(file_path + '\n')
    print(f"File paths saved to: selected_edgar_files.txt")
    
    return selected_files

def move_unselected_files(selected_files_csv: str = 'edgar_document_selection.csv', archive_folder: str = '../edgar-crawler/datasets/ARCHIVED_FILINGS'):

    
    if not os.path.exists(selected_files_csv):
        print(f"Error: {selected_files_csv} not found. Please run sample_edgar_filings() first.")
        return
    
    print("=" * 80)
    print("Moving Unselected Files to Archive")
    print("=" * 80)
    
    # Load the selected files from CSV
    df = pd.read_csv(selected_files_csv)
    selected_filenames = set(df['filename'].tolist())
    
    print(f"\nLoaded {len(selected_filenames)} selected files from CSV")
    
    # Define form types and paths
    edgar_base_path = "../edgar-crawler/datasets/RAW_FILINGS"
    form_types = ['10-K', '10-Q', '8-K']
    
    # Create archive folder structure
    os.makedirs(archive_folder, exist_ok=True)
    for form_type in form_types:
        os.makedirs(os.path.join(archive_folder, form_type), exist_ok=True)
    
    # Statistics
    moved_counts = defaultdict(int)
    total_moved = 0
    total_kept = 0
    
    # Process each form type
    for form_type in form_types:
        form_path = os.path.join(edgar_base_path, form_type)
        archive_form_path = os.path.join(archive_folder, form_type)
        
        if not os.path.exists(form_path):
            print(f"\nWarning: Directory not found: {form_path}")
            continue
        
        print(f"\nProcessing {form_type} files...")
        
        # Get all files in the directory
        all_files = [f for f in os.listdir(form_path) if f.endswith('.htm')]
        
        print(f"  Total files in directory: {len(all_files)}")
        
        # Move unselected files
        for filename in all_files:
            if filename not in selected_filenames:
                source_path = os.path.join(form_path, filename)
                dest_path = os.path.join(archive_form_path, filename)
                
                try:
                    shutil.move(source_path, dest_path)
                    moved_counts[form_type] += 1
                    total_moved += 1
                except Exception as e:
                    print(f"  Error moving {filename}: {e}")
            else:
                total_kept += 1
        
        print(f"  Moved: {moved_counts[form_type]} files")
        print(f"  Kept: {len(all_files) - moved_counts[form_type]} files")
    
    # Summary
    print("\n" + "=" * 80)
    print("Archive Summary")
    print("=" * 80)
    print(f"Total files moved to archive: {total_moved}")
    print(f"Total files kept in working directory: {total_kept}")
    print(f"\nArchive location: {os.path.abspath(archive_folder)}")
    
    # Breakdown by form type
    print("\nBreakdown by form type:")
    for form_type in form_types:
        print(f"  {form_type}: {moved_counts[form_type]} files archived")
    
    print("\nDone! Your working directory now contains only the selected files.")


def restore_archived_files(archive_folder: str = '../edgar-crawler/datasets/ARCHIVED_FILINGS'):

    import os
    import shutil
    from collections import defaultdict
    
    if not os.path.exists(archive_folder):
        print(f"Error: Archive folder not found: {archive_folder}")
        return
    
    print("=" * 80)
    print("Restoring Archived Files")
    print("=" * 80)
    
    edgar_base_path = "../edgar-crawler/datasets/RAW_FILINGS"
    form_types = ['10-K', '10-Q', '8-K']
    
    restored_counts = defaultdict(int)
    total_restored = 0
    
    # Process each form type
    for form_type in form_types:
        archive_form_path = os.path.join(archive_folder, form_type)
        form_path = os.path.join(edgar_base_path, form_type)
        
        if not os.path.exists(archive_form_path):
            continue
        
        print(f"\nRestoring {form_type} files...")
        
        # Get all files in archive
        archived_files = [f for f in os.listdir(archive_form_path) if f.endswith('.htm')]
        
        print(f"  Found {len(archived_files)} archived files")
        
        # Move files back
        for filename in archived_files:
            source_path = os.path.join(archive_form_path, filename)
            dest_path = os.path.join(form_path, filename)
            
            try:
                shutil.move(source_path, dest_path)
                restored_counts[form_type] += 1
                total_restored += 1
            except Exception as e:
                print(f"  Error restoring {filename}: {e}")
        
        print(f"  Restored: {restored_counts[form_type]} files")
    
    # Summary
    print("\n" + "=" * 80)
    print("Restore Summary")
    print("=" * 80)
    print(f"Total files restored: {total_restored}")
    
    print("\nBreakdown by form type:")
    for form_type in form_types:
        if restored_counts[form_type] > 0:
            print(f"  {form_type}: {restored_counts[form_type]} files")
    
    print("\nDone! All archived files have been restored.")


# Uncomment to run
# fetch_and_categorize_nasdaq100()
# stratified_sample_companies()
#sample_edgar_filings()

# After running sample_edgar_filings(), uncomment the line below to move unselected files
move_unselected_files()

# To restore archived files, uncomment:
# restore_archived_files()


