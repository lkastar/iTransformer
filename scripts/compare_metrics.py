import sys
import re
import os

# ANSI Colors
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def parse_log(filepath):
    """
    Parses a training log file.
    Returns a dict mapping setting_string -> experiment_data
    Duplicates are handled by keeping the last occurrence.
    """
    experiments = {}
    
    current_run_info = "无"
    
    # Regex to find setting line and extract seq_len (ft), label_len (sl), pred_len (ll)
    # Pattern looks for _ft followed by digits, checking surrounding structure
    # setting usually looks like: ..._ft{}_sl{}_ll{}_pl{}...
    setting_pattern = re.compile(r'_ft(\d+)_sl(\d+)_ll(\d+)_')
    
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return {}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return {}

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for RUN INFO
        if line.startswith("RUN INFO:"):
            # Extract the content after "RUN INFO:"
            current_run_info = line.split("RUN INFO:", 1)[1].strip()
            i += 1
            continue
            
        # Check for setting line
        match = setting_pattern.search(line)
        if match:
            seq_len = int(match.group(1))
            label_len = int(match.group(2))
            pred_len = int(match.group(3))
            
            setting_str = line
            
            # Extract data name and model name
            # Structure: {model_id}_{model}_{data}_{features}_ft...
            # We take the part before _ft
            prefix = line[:match.start()]
            parts = prefix.split('_')
            
            # Heuristic parsing based on run.py structure: model_id_model_data_features_ft...
            data_name = "Unknown"
            model_name = "Unknown"
            
            # We assume features is last, data is 2nd to last, model is 3rd to last.
            # This works for data names without underscores.
            if len(parts) >= 3:
                data_name = parts[-2]
                model_name = parts[-3]
            elif len(parts) >= 2:
                data_name = parts[-2]
                
            # Look ahead for metrics
            metrics_found = False
            mse = None
            mae = None
            
            # Scan next few lines for 'mse:'
            j = i + 1
            found_metrics_line = False
            while j < len(lines) and j < i + 10: # Lookahead limit
                m_line = lines[j].strip()
                if "mse:" in m_line and "mae:" in m_line:
                    # Parse metrics
                    # Expected: mse:0.37539565563201904, mae:0.3910726010799408
                    try:
                        parts_m = m_line.split(',')
                        for p in parts_m:
                            p = p.strip()
                            if p.startswith("mse:"):
                                mse = float(p.split(':')[1])
                            elif p.startswith("mae:"):
                                mae = float(p.split(':')[1])
                        
                        if mse is not None and mae is not None:
                            metrics_found = True
                            found_metrics_line = True
                    except:
                        pass
                    break
                
                # If we hit another RUN INFO or setting, stop
                if m_line.startswith("RUN INFO:") or setting_pattern.search(m_line):
                    break
                j += 1
            
            if metrics_found:
                # Store experiment
                experiments[setting_str] = {
                    'setting': setting_str,
                    'data': data_name,
                    'model': model_name,
                    'seq_len': seq_len,
                    'pred_len': pred_len,
                    'mse': mse,
                    'mae': mae,
                    'run_info': current_run_info
                }
            
            # Reset run info
            current_run_info = "无"
            
            if found_metrics_line:
                i = j  # Advance to the metrics line
            # continue loop will increment i at end or we can just proceed
        
        i += 1
        
    return experiments

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/compare_metrics.py <log_file_1> <log_file_2>")
        sys.exit(1)
        
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    exp1 = parse_log(file1)
    exp2 = parse_log(file2)
    
    # Helper to group by (data, seq_len, pred_len)
    def group_experiments(exp_dict):
        groups = {}
        for k, v in exp_dict.items():
            key = (v['data'], v['seq_len'], v['pred_len'])
            if key not in groups:
                groups[key] = []
            groups[key].append(v)
        return groups

    g1 = group_experiments(exp1)
    g2 = group_experiments(exp2)
    
    # Find intersection of configuration keys
    common_keys = set(g1.keys()) | set(g2.keys())
    
    if not common_keys:
        print("No configurations found.")
        return

    # Sort keys for consistent output
    sorted_keys = sorted(list(common_keys), key=lambda x: (x[0], x[1], x[2]))
    
    # Header
    print("-" * 155)
    print(f"{'Source':<15} | {'Data':<15} {'Seq':<6} {'Pred':<6} | {'MSE':<22} {'MAE':<22} | {'Remark (RUN INFO)'}")
    print("-" * 155)
    
    for key in sorted_keys:
        data, sl, pl = key
        
        entries1 = g1.get(key, [])
        entries2 = g2.get(key, [])
        all_entries = entries1 + entries2
        
        if not all_entries:
            continue

        # Determine best metrics in this group (min mse, min mae)
        min_mse = min(e['mse'] for e in all_entries)
        min_mae = min(e['mae'] for e in all_entries)
        
        def fmt(val, best_val):
            s = f"{val:.4f}"
            if len(all_entries) > 1 and val == best_val:
                 return f"{GREEN}{s:<10}{RESET}"
            return f"{s:<10}"

        # Print entries from file 1
        for e in entries1:
            mse_str = fmt(e['mse'], min_mse)
            mae_str = fmt(e['mae'], min_mae)
            print(f"{e['model']:<15} | {data:<15} {sl:<6} {pl:<6} | {mse_str:<22} {mae_str:<22} | {e['run_info']}")
            
        # Print entries from file 2
        for e in entries2:
            mse_str = fmt(e['mse'], min_mse)
            mae_str = fmt(e['mae'], min_mae)
            print(f"{e['model']:<15} | {data:<15} {sl:<6} {pl:<6} | {mse_str:<22} {mae_str:<22} | {e['run_info']}")
        
        print("-" * 155)

if __name__ == "__main__":
    main()
