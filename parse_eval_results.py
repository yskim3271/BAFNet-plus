#!/usr/bin/env python3
import re
import pandas as pd
import argparse
from tabulate import tabulate

def parse_log_file(log_file_path):
    # Initialize data structure to store results
    results = {}
    
    # Define regex patterns
    se_pattern = re.compile(r"Performance on (-?\d+)dB: PESQ=([0-9.]+), STOI=([0-9.]+), CSIG=([0-9.]+), CBAK=([0-9.]+), COVL=([0-9.]+)")
    stt_pattern = re.compile(r"Performance on (-?\d+)dB: CER=([0-9.]+), WER=([0-9.]+)")
    
    # Read log file
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # Extract speech enhancement (SE) results
    se_matches = se_pattern.finditer(log_content)
    for match in se_matches:
        snr = match.group(1) + "dB"
        
        if snr not in results:
            results[snr] = {}
            
        results[snr]["PESQ"] = float(match.group(2))
        results[snr]["STOI"] = float(match.group(3))
        results[snr]["CSIG"] = float(match.group(4))
        results[snr]["CBAK"] = float(match.group(5))
        results[snr]["COVL"] = float(match.group(6))
    
    # Extract speech recognition (STT) results
    stt_matches = stt_pattern.finditer(log_content)
    for match in stt_matches:
        snr = match.group(1) + "dB"
        
        if snr not in results:
            results[snr] = {}
            
        results[snr]["CER"] = float(match.group(2))
        results[snr]["WER"] = float(match.group(3))
    
    return results

def create_dataframe(results):
    # Initialize list to store data
    data = []
    
    # Convert results to list
    for snr in sorted(results.keys(), key=lambda x: int(x.replace("dB", ""))):
        metrics = results[snr]
        row = {"SNR": snr}
        row.update(metrics)
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Adjust column order
    columns = ["SNR"]
    se_metrics = ["PESQ", "STOI", "CSIG", "CBAK", "COVL"]
    stt_metrics = ["CER", "WER"]
    
    available_columns = columns + [col for col in se_metrics + stt_metrics if col in df.columns]
    df = df[available_columns]
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Parse evaluation results from trainer log file')
    parser.add_argument('log_file', type=str, help='Path to the trainer log file')
    parser.add_argument('--output', type=str, help='Output CSV file path (optional)')
    parser.add_argument('--format', type=str, default='csv', choices=['pretty', 'csv', 'markdown'],
                        help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Parse log file
    results = parse_log_file(args.log_file)
    
    # Create DataFrame
    df = create_dataframe(results)
    
    # Output results
    if args.format == 'pretty':
        print(tabulate(df, headers='keys', tablefmt='pretty', floatfmt='.4f'))
    elif args.format == 'markdown':
        print(tabulate(df, headers='keys', tablefmt='github', floatfmt='.4f'))
    elif args.format == 'csv':
        print(df.to_csv(index=False))
    
    # Save results (if requested)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main() 