#!/usr/bin/env python3
"""
Training Log Analysis Script
Analyzes GPT-2 training logs from Single_GPU_12hrs_train folder
Extracts model parameters, training metrics, and creates visualizations
"""

import re
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class TrainingLogAnalyzer:
    def __init__(self, log_dir="Single_GPU_12hrs_train"):
        self.log_dir = log_dir
        self.log_files = sorted(glob.glob(os.path.join(log_dir, "*.o*")))
        self.experiments = []
        
    def extract_config(self, filepath, max_lines=500):
        """Extract configuration from log file"""
        config = {
            'file': os.path.basename(filepath),
            'job_id': None,
            'flash_attention': False,
            'sequence_length': None,
            'num_parameters': None,
            'num_layers': None,
            'channels': None,
            'batch_size': None,
            'learning_rate': None,
            'device': None,
            'gpu_memory': None,
            'max_seq_len': None,
            'val_losses': [],
            'train_losses': [],
            'steps': [],
            'mfu': [],
            'throughput': []
        }
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Extract job ID from filename
        match = re.search(r'\.o(\d+)$', filepath)
        if match:
            config['job_id'] = match.group(1)
        
        # Parse first max_lines for config
        for i, line in enumerate(lines[:max_lines]):
            # Check for Flash Attention
            if 'cuDNN Flash Attention' in line or 'ENABLE_CUDNN' in line:
                config['flash_attention'] = True
            if 'standard attention' in line.lower() or 'cuDNN disabled' in line:
                config['flash_attention'] = False
                
            # Extract parameters
            if '| sequence length T' in line:
                match = re.search(r'\|\s*(\d+)', line.split('|')[2])
                if match:
                    config['sequence_length'] = int(match.group(1))
                    
            if '| num_parameters' in line:
                match = re.search(r'\|\s*(\d+)', line.split('|')[2])
                if match:
                    config['num_parameters'] = int(match.group(1))
                    
            if '| num_layers L' in line:
                match = re.search(r'\|\s*(\d+)', line.split('|')[2])
                if match:
                    config['num_layers'] = int(match.group(1))
                    
            if '| channels C' in line:
                match = re.search(r'\|\s*(\d+)', line.split('|')[2])
                if match:
                    config['channels'] = int(match.group(1))
                    
            if '| micro batch size B' in line:
                match = re.search(r'\|\s*(\d+)', line.split('|')[2])
                if match:
                    config['batch_size'] = int(match.group(1))
                    
            if '| learning rate (LR)' in line:
                match = re.search(r'([\d.e\-+]+)', line.split('|')[2])
                if match:
                    config['learning_rate'] = float(match.group(1))
                    
            if '| device' in line and 'NVIDIA' in line:
                device = line.split('|')[2].strip()
                config['device'] = device
                # Extract memory size
                mem_match = re.search(r'(\d+)GB', device)
                if mem_match:
                    config['gpu_memory'] = int(mem_match.group(1))
                    
            if '| max_sequence_length T' in line:
                match = re.search(r'\|\s*(\d+)', line.split('|')[2])
                if match:
                    config['max_seq_len'] = int(match.group(1))
        
        # Parse training metrics from entire file
        for line in lines:
            # Extract validation loss
            if line.strip().startswith('val loss'):
                match = re.search(r'val loss\s+([\d.]+)', line)
                if match:
                    config['val_losses'].append(float(match.group(1)))
                    
            # Extract training step info
            step_match = re.search(r'step\s+(\d+)/\d+\s*\|\s*loss\s+([\d.]+).*?\|\s*.*?(\d+\.?\d*)%\s*bf16\s*MFU\s*\|\s*(\d+)\s*tok/s', line)
            if step_match:
                step = int(step_match.group(1))
                loss = float(step_match.group(2))
                mfu = float(step_match.group(3))
                throughput = int(step_match.group(4))
                
                config['steps'].append(step)
                config['train_losses'].append(loss)
                config['mfu'].append(mfu)
                config['throughput'].append(throughput)
        
        return config
    
    def analyze_all_logs(self):
        """Analyze all log files"""
        print(f"Found {len(self.log_files)} log files")
        
        for log_file in self.log_files:
            print(f"Processing: {os.path.basename(log_file)}")
            config = self.extract_config(log_file)
            if config['sequence_length'] is not None:
                self.experiments.append(config)
                
        print(f"\nSuccessfully parsed {len(self.experiments)} experiments")
        return self.experiments
    
    def create_summary_table(self):
        """Create summary table of all experiments"""
        summary_data = []
        
        for exp in self.experiments:
            summary_data.append({
                'Job ID': exp['job_id'],
                'Sequence Length': exp['sequence_length'],
                'Parameters (M)': f"{exp['num_parameters']/1e6:.1f}" if exp['num_parameters'] else 'N/A',
                'Flash Attention': 'Yes' if exp['flash_attention'] else 'No',
                'GPU Memory (GB)': exp['gpu_memory'],
                'Layers': exp['num_layers'],
                'Channels': exp['channels'],
                'Initial Val Loss': f"{exp['val_losses'][0]:.3f}" if exp['val_losses'] else 'N/A',
                'Final Val Loss': f"{exp['val_losses'][-1]:.3f}" if exp['val_losses'] else 'N/A',
                'Total Steps': len(exp['train_losses']),
                'Avg MFU (%)': f"{np.mean(exp['mfu']):.1f}" if exp['mfu'] else 'N/A',
                'Avg Throughput (tok/s)': f"{np.mean(exp['throughput']):.0f}" if exp['throughput'] else 'N/A'
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('Sequence Length')
        return df
    
    def plot_training_curves(self, save_path='training_loss_curves.png'):
        """Plot training loss curves for all experiments"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Group by Flash Attention
        flash_exps = [e for e in self.experiments if e['flash_attention']]
        std_exps = [e for e in self.experiments if not e['flash_attention']]
        
        # Plot Flash Attention experiments
        for exp in flash_exps:
            if exp['train_losses']:
                label = f"Seq={exp['sequence_length']}, FA"
                ax1.plot(exp['steps'][:500], exp['train_losses'][:500], 
                        marker='o', markersize=2, linewidth=1.5, alpha=0.7, label=label)
        
        ax1.set_xlabel('Training Step', fontsize=12)
        ax1.set_ylabel('Training Loss', fontsize=12)
        ax1.set_title('Training Loss: Flash Attention (cuDNN)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot Standard Attention experiments
        for exp in std_exps:
            if exp['train_losses']:
                label = f"Seq={exp['sequence_length']}, Std"
                ax2.plot(exp['steps'][:500], exp['train_losses'][:500], 
                        marker='s', markersize=2, linewidth=1.5, alpha=0.7, label=label)
        
        ax2.set_xlabel('Training Step', fontsize=12)
        ax2.set_ylabel('Training Loss', fontsize=12)
        ax2.set_title('Training Loss: Standard Attention', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_validation_loss(self, save_path='validation_loss_comparison.png'):
        """Plot validation loss progression"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.experiments)))
        
        for i, exp in enumerate(self.experiments):
            if exp['val_losses']:
                attention_type = "Flash" if exp['flash_attention'] else "Standard"
                label = f"Seq={exp['sequence_length']}, {attention_type}"
                eval_points = list(range(0, len(exp['val_losses']) * 500, 500))
                ax.plot(eval_points, exp['val_losses'], 
                       marker='o', markersize=6, linewidth=2, 
                       color=colors[i], label=label, alpha=0.8)
        
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=12)
        ax.set_title('Validation Loss Progression Across Configurations', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_performance_metrics(self, save_path='performance_metrics.png'):
        """Plot MFU and throughput comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Prepare data
        seq_lengths = []
        mfu_flash = []
        mfu_std = []
        throughput_flash = []
        throughput_std = []
        
        for exp in self.experiments:
            if exp['mfu'] and exp['throughput']:
                seq_len = exp['sequence_length']
                avg_mfu = np.mean(exp['mfu'])
                avg_throughput = np.mean(exp['throughput'])
                
                if exp['flash_attention']:
                    if seq_len not in seq_lengths:
                        seq_lengths.append(seq_len)
                        mfu_flash.append(avg_mfu)
                        throughput_flash.append(avg_throughput)
                    else:
                        idx = seq_lengths.index(seq_len)
                        mfu_flash[idx] = max(mfu_flash[idx], avg_mfu)
                        throughput_flash[idx] = max(throughput_flash[idx], avg_throughput)
                else:
                    if seq_len not in seq_lengths:
                        seq_lengths.append(seq_len)
                        mfu_std.append(avg_mfu)
                        throughput_std.append(avg_throughput)
                    else:
                        idx = seq_lengths.index(seq_len)
                        mfu_std[idx] = max(mfu_std[idx], avg_mfu)
                        throughput_std[idx] = max(throughput_std[idx], avg_throughput)
        
        # Sort by sequence length
        sorted_indices = np.argsort(seq_lengths)
        seq_lengths = [seq_lengths[i] for i in sorted_indices]
        
        # MFU comparison
        x = np.arange(len(seq_lengths))
        width = 0.35
        
        if mfu_flash:
            ax1.bar(x - width/2, [mfu_flash[seq_lengths.index(s)] if s in [exp['sequence_length'] for exp in self.experiments if exp['flash_attention']] else 0 for s in seq_lengths], 
                   width, label='Flash Attention', color='#2ecc71', alpha=0.8)
        if mfu_std:
            ax1.bar(x + width/2, [mfu_std[seq_lengths.index(s)] if s in [exp['sequence_length'] for exp in self.experiments if not exp['flash_attention']] else 0 for s in seq_lengths], 
                   width, label='Standard Attention', color='#3498db', alpha=0.8)
        
        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Model FLOPs Utilization (%)', fontsize=12)
        ax1.set_title('MFU Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(seq_lengths)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Throughput comparison
        if throughput_flash:
            ax2.bar(x - width/2, [throughput_flash[seq_lengths.index(s)] if s in [exp['sequence_length'] for exp in self.experiments if exp['flash_attention']] else 0 for s in seq_lengths], 
                   width, label='Flash Attention', color='#2ecc71', alpha=0.8)
        if throughput_std:
            ax2.bar(x + width/2, [throughput_std[seq_lengths.index(s)] if s in [exp['sequence_length'] for exp in self.experiments if not exp['flash_attention']] else 0 for s in seq_lengths], 
                   width, label='Standard Attention', color='#3498db', alpha=0.8)
        
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax2.set_title('Training Throughput Comparison', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(seq_lengths)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def plot_sequence_length_impact(self, save_path='sequence_length_impact.png'):
        """Plot impact of sequence length on various metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Organize data by sequence length
        seq_data = {}
        for exp in self.experiments:
            seq_len = exp['sequence_length']
            att_type = 'Flash' if exp['flash_attention'] else 'Standard'
            
            if seq_len not in seq_data:
                seq_data[seq_len] = {'Flash': {}, 'Standard': {}}
            
            if exp['val_losses']:
                seq_data[seq_len][att_type]['final_val_loss'] = exp['val_losses'][-1]
            if exp['mfu']:
                seq_data[seq_len][att_type]['avg_mfu'] = np.mean(exp['mfu'])
            if exp['throughput']:
                seq_data[seq_len][att_type]['avg_throughput'] = np.mean(exp['throughput'])
            if exp['num_parameters']:
                seq_data[seq_len][att_type]['num_params'] = exp['num_parameters'] / 1e6
        
        # Plot 1: Final validation loss vs sequence length
        for att_type, color in [('Flash', '#2ecc71'), ('Standard', '#3498db')]:
            seq_lens = []
            val_losses = []
            for seq_len in sorted(seq_data.keys()):
                if 'final_val_loss' in seq_data[seq_len][att_type]:
                    seq_lens.append(seq_len)
                    val_losses.append(seq_data[seq_len][att_type]['final_val_loss'])
            if seq_lens:
                ax1.plot(seq_lens, val_losses, marker='o', markersize=10, 
                        linewidth=2, label=f'{att_type} Attention', color=color)
        
        ax1.set_xlabel('Sequence Length', fontsize=12)
        ax1.set_ylabel('Final Validation Loss', fontsize=12)
        ax1.set_title('Model Quality vs Sequence Length', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MFU vs sequence length
        for att_type, color in [('Flash', '#2ecc71'), ('Standard', '#3498db')]:
            seq_lens = []
            mfus = []
            for seq_len in sorted(seq_data.keys()):
                if 'avg_mfu' in seq_data[seq_len][att_type]:
                    seq_lens.append(seq_len)
                    mfus.append(seq_data[seq_len][att_type]['avg_mfu'])
            if seq_lens:
                ax2.plot(seq_lens, mfus, marker='s', markersize=10, 
                        linewidth=2, label=f'{att_type} Attention', color=color)
        
        ax2.set_xlabel('Sequence Length', fontsize=12)
        ax2.set_ylabel('Average MFU (%)', fontsize=12)
        ax2.set_title('Hardware Efficiency vs Sequence Length', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Throughput vs sequence length
        for att_type, color in [('Flash', '#2ecc71'), ('Standard', '#3498db')]:
            seq_lens = []
            throughputs = []
            for seq_len in sorted(seq_data.keys()):
                if 'avg_throughput' in seq_data[seq_len][att_type]:
                    seq_lens.append(seq_len)
                    throughputs.append(seq_data[seq_len][att_type]['avg_throughput'])
            if seq_lens:
                ax3.plot(seq_lens, throughputs, marker='^', markersize=10, 
                        linewidth=2, label=f'{att_type} Attention', color=color)
        
        ax3.set_xlabel('Sequence Length', fontsize=12)
        ax3.set_ylabel('Throughput (tokens/sec)', fontsize=12)
        ax3.set_title('Training Speed vs Sequence Length', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model size vs sequence length
        for att_type, color in [('Flash', '#2ecc71'), ('Standard', '#3498db')]:
            seq_lens = []
            params = []
            for seq_len in sorted(seq_data.keys()):
                if 'num_params' in seq_data[seq_len][att_type]:
                    seq_lens.append(seq_len)
                    params.append(seq_data[seq_len][att_type]['num_params'])
            if seq_lens:
                ax4.plot(seq_lens, params, marker='D', markersize=10, 
                        linewidth=2, label=f'{att_type} Attention', color=color)
        
        ax4.set_xlabel('Sequence Length', fontsize=12)
        ax4.set_ylabel('Parameters (Millions)', fontsize=12)
        ax4.set_title('Model Size vs Sequence Length', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def generate_ppt_summary(self, output_file='training_summary.txt'):
        """Generate text summary suitable for PPT slides"""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GPT-2 TRAINING EXPERIMENTS SUMMARY\n")
            f.write("Single GPU Training - 12 Hour Runs\n")
            f.write("=" * 80 + "\n\n")
            
            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n\n")
            
            # Flash Attention Impact
            flash_exps = [e for e in self.experiments if e['flash_attention'] and e['mfu']]
            std_exps = [e for e in self.experiments if not e['flash_attention'] and e['mfu']]
            
            if flash_exps and std_exps:
                avg_mfu_flash = np.mean([np.mean(e['mfu']) for e in flash_exps])
                avg_mfu_std = np.mean([np.mean(e['mfu']) for e in std_exps])
                speedup = (avg_mfu_flash / avg_mfu_std - 1) * 100
                
                f.write(f"1. FLASH ATTENTION PERFORMANCE:\n")
                f.write(f"   - Flash Attention Average MFU: {avg_mfu_flash:.1f}%\n")
                f.write(f"   - Standard Attention Average MFU: {avg_mfu_std:.1f}%\n")
                f.write(f"   - Performance Improvement: {speedup:+.1f}%\n\n")
                
                avg_throughput_flash = np.mean([np.mean(e['throughput']) for e in flash_exps])
                avg_throughput_std = np.mean([np.mean(e['throughput']) for e in std_exps])
                
                f.write(f"   - Flash Attention Throughput: {avg_throughput_flash:.0f} tokens/sec\n")
                f.write(f"   - Standard Attention Throughput: {avg_throughput_std:.0f} tokens/sec\n")
                f.write(f"   - Throughput Improvement: {(avg_throughput_flash/avg_throughput_std - 1)*100:+.1f}%\n\n")
            
            # Sequence Length Impact
            f.write("2. SEQUENCE LENGTH IMPACT:\n")
            seq_lengths = sorted(list(set([e['sequence_length'] for e in self.experiments if e['sequence_length']])))
            for seq_len in seq_lengths:
                exps_at_len = [e for e in self.experiments if e['sequence_length'] == seq_len]
                if exps_at_len and exps_at_len[0]['val_losses']:
                    avg_final_loss = np.mean([e['val_losses'][-1] for e in exps_at_len if e['val_losses']])
                    f.write(f"   - Seq Length {seq_len}: Final Val Loss = {avg_final_loss:.3f}\n")
            f.write("\n")
            
            # Model Configurations
            f.write("3. MODEL CONFIGURATIONS TESTED:\n")
            for i, exp in enumerate(self.experiments, 1):
                f.write(f"\n   Config {i} (Job {exp['job_id']}):\n")
                f.write(f"   - Sequence Length: {exp['sequence_length']}\n")
                f.write(f"   - Parameters: {exp['num_parameters']/1e6:.1f}M\n")
                f.write(f"   - Layers: {exp['num_layers']}, Channels: {exp['channels']}\n")
                f.write(f"   - Flash Attention: {'Yes' if exp['flash_attention'] else 'No'}\n")
                f.write(f"   - GPU: {exp['device']}\n")
                if exp['val_losses']:
                    f.write(f"   - Validation Loss: {exp['val_losses'][0]:.3f} → {exp['val_losses'][-1]:.3f}\n")
                if exp['mfu']:
                    f.write(f"   - Average MFU: {np.mean(exp['mfu']):.1f}%\n")
                if exp['throughput']:
                    f.write(f"   - Average Throughput: {np.mean(exp['throughput']):.0f} tokens/sec\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS FOR PRESENTATION:\n")
            f.write("=" * 80 + "\n\n")
            f.write("1. Highlight Flash Attention efficiency gains\n")
            f.write("2. Show impact of sequence length on model quality and speed\n")
            f.write("3. Demonstrate trade-offs between sequence length and throughput\n")
            f.write("4. Compare GPU memory requirements across configurations\n")
            f.write("5. Emphasize reproducibility with different model sizes\n\n")
        
        print(f"Saved: {output_file}")
    
    def save_data_to_csv(self, output_file='training_data.csv'):
        """Save extracted data to CSV for further analysis"""
        all_data = []
        
        for exp in self.experiments:
            base_row = {
                'job_id': exp['job_id'],
                'sequence_length': exp['sequence_length'],
                'num_parameters': exp['num_parameters'],
                'flash_attention': exp['flash_attention'],
                'num_layers': exp['num_layers'],
                'channels': exp['channels'],
                'gpu_memory': exp['gpu_memory'],
                'device': exp['device']
            }
            
            # Add training data
            for i, step in enumerate(exp['steps']):
                row = base_row.copy()
                row['step'] = step
                row['train_loss'] = exp['train_losses'][i] if i < len(exp['train_losses']) else None
                row['mfu'] = exp['mfu'][i] if i < len(exp['mfu']) else None
                row['throughput'] = exp['throughput'][i] if i < len(exp['throughput']) else None
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        return df


def main():
    print("=" * 80)
    print("GPT-2 TRAINING LOG ANALYSIS")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    analyzer = TrainingLogAnalyzer()
    
    # Analyze logs
    print("Step 1: Extracting data from log files...")
    experiments = analyzer.analyze_all_logs()
    
    if not experiments:
        print("❌ No experiments found!")
        return
    
    print(f"\n✅ Successfully analyzed {len(experiments)} experiments\n")
    
    # Create summary table
    print("Step 2: Creating summary table...")
    summary_df = analyzer.create_summary_table()
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save summary table
    summary_df.to_csv('experiment_summary.csv', index=False)
    print("✅ Saved: experiment_summary.csv\n")
    
    # Generate visualizations
    print("Step 3: Generating visualizations...")
    analyzer.plot_training_curves()
    analyzer.plot_validation_loss()
    analyzer.plot_performance_metrics()
    analyzer.plot_sequence_length_impact()
    
    # Generate text summary
    print("\nStep 4: Generating PPT summary...")
    analyzer.generate_ppt_summary()
    
    # Save detailed data
    print("\nStep 5: Saving detailed data to CSV...")
    analyzer.save_data_to_csv()
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  📊 training_loss_curves.png")
    print("  📊 validation_loss_comparison.png")
    print("  📊 performance_metrics.png")
    print("  📊 sequence_length_impact.png")
    print("  📄 experiment_summary.csv")
    print("  📄 training_summary.txt")
    print("  📄 training_data.csv")
    print("\nThese files are ready for your PPT presentation!")
    print("=" * 80)


if __name__ == "__main__":
    main()
