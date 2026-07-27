#!/usr/bin/env python3
"""
Collect FC Results
==================
Collects results from all FC test runs and updates FC_results.csv
"""

import json
import csv
from pathlib import Path

RUN6_DIR = Path(__file__).parent

BASE_MODELS = [
    'convnext', 'dat', 'drct', 'edsr', 'gmsr', 'hat', 'hdnet', 'hrnet',
    'hscnn', 'hsrmamba', 'mirnet', 'mprnet', 'mst', 'nafnet', 'padut',
    'pix2pix', 'rdn', 'restormer', 'retinexformer', 'sr3', 'swin2sr',
    'swinir', 'unetpp', 'vmambair'
]

MODEL_RUNS = [
    {
        'model_key': f'{m}_fc',
        'model_dir': f'phamscope_{m}',
        'summary_rel': 'test_results_fc/summary.json',
        'log_rel': f'logs_fc/{m}_fc_training_log.json',
        'train_script': f'train_{m}_fc.py',
        'test_script': f'test_{m}_fc.py',
    }
    for m in BASE_MODELS
]

# SPECAT variants are run in one folder with explicit script/summary names.
MODEL_RUNS.extend([
    {
        'model_key': 'specat_s1_fc',
        'model_dir': 'phamscope_specat',
        'summary_rel': 'test_results_fc_specat_s1/summary.json',
        'log_rel': 'logs_fc/specat_s1_fc_training_log.json',
        'train_script': 'train_specat_s1_fc.py',
        'test_script': 'test_specat_s1_fc.py',
    },
    {
        'model_key': 'specat_s2_fc',
        'model_dir': 'phamscope_specat',
        'summary_rel': 'test_results_fc_specat_s2/summary.json',
        'log_rel': 'logs_fc/specat_s2_fc_training_log.json',
        'train_script': 'train_specat_s2_fc.py',
        'test_script': 'test_specat_s2_fc.py',
    },
    {
        'model_key': 'specat_realmask_l1_fc_s1',
        'model_dir': 'phamscope_specat',
        'summary_rel': 'test_results_realmask_l1_fc_s1/summary.json',
        'log_rel': 'logs_realmask_l1_fc_s1/specat_realmask_l1_fc_s1_training_log.json',
        'train_script': 'train_specat_realmask_l1_fc_s1.py',
        'test_script': 'test_specat_realmask_l1_fc_s1.py',
    },
    {
        'model_key': 'specat_realmask_l1_fc_s2',
        'model_dir': 'phamscope_specat',
        'summary_rel': 'test_results_realmask_l1_fc_s2/summary.json',
        'log_rel': 'logs_realmask_l1_fc_s2/specat_realmask_l1_fc_s2_training_log.json',
        'train_script': 'train_specat_realmask_l1_fc_s2.py',
        'test_script': 'test_specat_realmask_l1_fc_s2.py',
    },
    {
        'model_key': 'specat_realmask_l1_s1_synth_optcal_10e_ft_fc',
        'model_dir': 'phamscope_specat',
        'summary_rel': 'test_results_realmask_l1_s1_synth_optcal_10e_ft_fc/summary.json',
        'log_rel': 'logs_realmask_l1_s1_synth_optcal_10e_ft_fc/specat_realmask_l1_s1_synth_optcal_10e_ft_fc_training_log.json',
        'train_script': 'train_specat_realmask_l1_s1_synth_optcal_ft_fc.py',
        'test_script': 'test_specat_realmask_l1_s1_synth_optcal_ft_fc.py',
    },
    {
        'model_key': 'specat_realmask_l1_s2_synth_optcal_10e_ft_fc',
        'model_dir': 'phamscope_specat',
        'summary_rel': 'test_results_realmask_l1_s2_synth_optcal_10e_ft_fc/summary.json',
        'log_rel': 'logs_realmask_l1_s2_synth_optcal_10e_ft_fc/specat_realmask_l1_s2_synth_optcal_10e_ft_fc_training_log.json',
        'train_script': 'train_specat_realmask_l1_s2_synth_optcal_ft_fc.py',
        'test_script': 'test_specat_realmask_l1_s2_synth_optcal_ft_fc.py',
    },
    {
        'model_key': 'reggan_fc',
        'model_dir': 'phamscope_reggan',
        'summary_rel': 'test_results_fc_reggan/summary.json',
        'log_rel': 'logs_fc/reggan_fc_training_log.json',
        'train_script': 'train_reggan_fc.py',
        'test_script': 'test_reggan_fc.py',
    },
    {
        'model_key': 'scope_fc',
        'model_dir': 'phamscope_bassai',
        'summary_rel': 'test_results_scope_fc/summary.json',
        'log_rel': 'logs_scope_fc/scope_fc_training_log.json',
        'train_script': 'train_scope_fc.py --mode synth --epochs 10 ; train_scope_fc.py --mode ft --variant scope_fc --epochs 10',
        'test_script': 'test_scope_fc.py --checkpoint-mode ft --eval-split real_raw',
    },
    {
        'model_key': 'scope_fc_ft20',
        'model_dir': 'phamscope_bassai',
        'summary_rel': 'test_results_scope_fc_ft20/summary.json',
        'log_rel': 'logs_scope_fc_ft20/scope_fc_ft20_training_log.json',
        'train_script': 'train_scope_fc.py --mode synth --epochs 10 ; train_scope_fc.py --mode ft --variant scope_fc_ft20 --epochs 20',
        'test_script': 'test_scope_fc.py --checkpoint-mode ft --variant scope_fc_ft20 --eval-split real_raw',
    },
    {
        'model_key': 'scope_fc_ft20_nocropshift',
        'model_dir': 'phamscope_bassai',
        'summary_rel': 'test_results_scope_fc_ft20_nocropshift/summary.json',
        'log_rel': 'logs_scope_fc_ft20_nocropshift/scope_fc_ft20_nocropshift_training_log.json',
        'train_script': 'train_scope_fc.py --mode synth --epochs 10 ; train_scope_fc.py --mode ft --variant scope_fc_ft20_nocropshift --epochs 20',
        'test_script': 'test_scope_fc.py --checkpoint-mode ft --variant scope_fc_ft20_nocropshift --eval-split real_raw',
    },
])


def load_existing_rows(csv_path: Path) -> dict:
    """Load existing CSV rows keyed by Model.

    This allows incremental updates without wiping pre-filled fields like Params_M.
    """
    if not csv_path.exists():
        return {}

    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        return {row.get('Model', ''): row for row in reader if row.get('Model')}


def collect_results():
    """Collect results from all FC test runs."""
    results = []

    csv_path = RUN6_DIR / 'FC_results.csv'
    existing = load_existing_rows(csv_path)
    
    for run in MODEL_RUNS:
        model_dir = RUN6_DIR / run['model_dir']
        summary_path = model_dir / run['summary_rel']
        log_path = model_dir / run['log_rel']
        model_key = run['model_key']
        prev = existing.get(model_key, {})
        
        row = {
            'Model': model_key,
            # Preserve any pre-filled values from the existing CSV
            'Params_M': prev.get('Params_M', ''),
            'Test_MAE': prev.get('Test_MAE', ''),
            'Test_MSE': prev.get('Test_MSE', ''),
            'Test_PSNR': prev.get('Test_PSNR', ''),
            'Test_SSIM_2D': prev.get('Test_SSIM_2D', ''),
            'Test_SSIM_3D': prev.get('Test_SSIM_3D', ''),
            'Inference_Time_s': prev.get('Inference_Time_s', ''),
            'Train_Loss': prev.get('Train_Loss', ''),
            'Train_Time_h': prev.get('Train_Time_h', ''),
            'Epochs': prev.get('Epochs', '20') or '20',
            'Batch_Size': prev.get('Batch_Size', '2') or '2',
            'Patch_Size': prev.get('Patch_Size', ''),
            'Training_Script': run['train_script'],
            'Test_Script': run['test_script'],
        }
        
        # Load test results
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            
            row['Test_MAE'] = f"{summary.get('mae', ''):.6f}" if summary.get('mae') else ''
            row['Test_MSE'] = f"{summary.get('mse', ''):.6f}" if summary.get('mse') else ''
            row['Test_PSNR'] = f"{summary.get('psnr', ''):.2f}" if summary.get('psnr') else ''
            row['Test_SSIM_2D'] = f"{summary.get('ssim_2d', ''):.4f}" if summary.get('ssim_2d') else ''
            row['Test_SSIM_3D'] = f"{summary.get('ssim_3d', ''):.4f}" if summary.get('ssim_3d') else ''
            row['Inference_Time_s'] = f"{summary.get('inference_time', ''):.4f}" if summary.get('inference_time') else ''
        
        # Load training log
        if log_path.exists():
            with open(log_path, 'r') as f:
                log = json.load(f)
            
            row['Train_Loss'] = f"{log.get('final_train_loss', ''):.6f}" if log.get('final_train_loss') else ''
            
            # Convert seconds to hours
            total_time = log.get('total_time_seconds', 0)
            if total_time:
                row['Train_Time_h'] = f"{total_time / 3600:.2f}"
            
            # Get params from config
            config = log.get('config', {})
            if config.get('num_params'):
                row['Params_M'] = f"{config['num_params'] / 1e6:.2f}"
            if config.get('batch_size'):
                row['Batch_Size'] = str(config['batch_size'])
            if config.get('epochs_label'):
                row['Epochs'] = str(config['epochs_label'])
            elif config.get('num_epochs'):
                row['Epochs'] = str(config['num_epochs'])
            if config.get('use_patches'):
                row['Patch_Size'] = str(config.get('patch_size') or '')
            else:
                row['Patch_Size'] = ''
            if config.get('training_script'):
                row['Training_Script'] = str(config['training_script'])

        if model_key == 'scope_fc' and row['Epochs'] in ('', '10'):
            row['Epochs'] = '10 synth + 10 ft'
        elif model_key in {'scope_fc_ft20', 'scope_fc_ft20_nocropshift'} and row['Epochs'] in ('', '20'):
            row['Epochs'] = '10 synth + 20 ft'

        results.append(row)
        print(f"  {model_key}: MAE={row['Test_MAE'] or 'N/A'}, PSNR={row['Test_PSNR'] or 'N/A'}")
    
    return results


def save_results(results):
    """Save results to FC_results.csv."""
    output_path = RUN6_DIR / 'FC_results.csv'
    
    fieldnames = [
        'Model', 'Params_M', 'Test_MAE', 'Test_MSE', 'Test_PSNR',
        'Test_SSIM_2D', 'Test_SSIM_3D', 'Inference_Time_s', 'Train_Loss',
        'Train_Time_h', 'Epochs', 'Batch_Size', 'Patch_Size',
        'Training_Script', 'Test_Script'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {output_path}")


def main():
    print("=" * 70)
    print("Collecting FC Results")
    print("=" * 70)
    
    results = collect_results()
    save_results(results)
    
    # Summary statistics
    completed = sum(1 for r in results if r['Test_MAE'])
    print(f"\n{completed}/{len(results)} models completed")


if __name__ == '__main__':
    main()
