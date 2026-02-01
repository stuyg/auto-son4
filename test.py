import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def smooth_curve(x, y, num_points=500):
    """
    Use spline interpolation to generate smooth curves
    """
    # Simple sorting and deduplication
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    _, unique_indices = np.unique(x, return_index=True)
    x = x[unique_indices]
    y = y[unique_indices]

    if len(x) < 4: return x, y
    
    # k=3 cubic spline interpolation
    spl = make_interp_spline(x, y, k=3) 
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = spl(x_new)
    y_new = np.clip(y_new, 0, 1) # Limit to 0-1
    return x_new, y_new

def run_plot():
    # Set global font style similar to Image 2
    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(18, 8)) 
    
    # Base X-axis points (FPR)
    x_base = np.array([0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # Style mapping configuration (Color, Marker, LineStyle)
    styles = {
        'CNN-GCN':  {'c': '#FFA500', 'm': 'v', 'ls': '-',  'ms': 8, 'markevery': 40}, # Orange
        'GCN-CSS':  {'c': '#008000', 'm': 's', 'ls': '--', 'ms': 7, 'markevery': 40}, # Green
        'CNN':      {'c': '#FF0000', 'm': 'o', 'ls': '-',  'ms': 7, 'markevery': 40}, # Red
        'GAT-Attn': {'c': '#FF00FF', 'm': '*', 'ls': '--', 'ms': 10, 'markevery': 40}, # Magenta
        'MLP':      {'c': '#0000FF', 'm': 'd', 'ls': '-',  'ms': 7, 'markevery': 40}, # Blue
    }

    # Data for SNR = -10dB
    data_10db = [
        ("CNN-GCN (Hybrid)", 
         [0.0, 0.30, 0.55, 0.75, 0.88, 0.93, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0], "0.8968"),
        
        ("GCN-CSS (Proposed)", 
         [0.0, 0.20, 0.40, 0.60, 0.78, 0.86, 0.91, 0.94, 0.97, 0.99, 1.0, 1.0, 1.0], "0.8375"),

        ("CNN", 
         [0.0, 0.15, 0.30, 0.48, 0.65, 0.75, 0.82, 0.88, 0.92, 0.96, 0.98, 1.0, 1.0], "0.7506"),

        ("GAT-Attn (New)", 
         [0.0, 0.10, 0.22, 0.38, 0.55, 0.65, 0.72, 0.80, 0.86, 0.92, 0.96, 0.99, 1.0], "0.7136"),

        ("MLP", 
         [0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97, 1.0], "0.6015"),
    ]

    # Data for SNR = -8dB
    data_8db = [
        ("CNN-GCN (Hybrid)", 
         [0.0, 0.50, 0.80, 0.92, 0.97, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "0.9739"),
        
        ("GCN-CSS (Proposed)", 
         [0.0, 0.40, 0.70, 0.85, 0.93, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0], "0.9435"),

        ("CNN", 
         [0.0, 0.25, 0.55, 0.75, 0.85, 0.90, 0.94, 0.96, 0.98, 0.99, 1.0, 1.0, 1.0], "0.8975"),

        ("GAT-Attn (New)", 
         [0.0, 0.20, 0.45, 0.65, 0.80, 0.88, 0.92, 0.95, 0.97, 0.99, 1.0, 1.0, 1.0], "0.8596"),

        ("MLP", 
         [0.0, 0.15, 0.30, 0.45, 0.55, 0.65, 0.72, 0.80, 0.88, 0.94, 0.98, 1.0, 1.0], "0.6543"),
    ]

    datasets = [(axes[0], data_10db, "-10dB"), (axes[1], data_8db, "-8dB")]

    for ax, dataset, snr_label in datasets:
        ax.grid(which='major', linestyle='-', linewidth='1.0', color='darkgray', alpha=0.5)
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray', alpha=0.3)
        ax.minorticks_on()

        for name_full, y_vals, auc_score in dataset:
            x_smooth, y_smooth = smooth_curve(x_base, np.array(y_vals), num_points=400)
            
            key = next((k for k in styles.keys() if k in name_full), 'MLP')
            st = styles[key]
            
            # Construct label with AUC
            label_text = f"{name_full} (AUC={auc_score})"
            
            ax.plot(x_smooth, y_smooth, 
                    label=label_text, 
                    color=st['c'], 
                    linestyle=st['ls'], 
                    linewidth=2.5,
                    marker=st['m'],
                    markersize=st['ms'],
                    markevery=st['markevery'], 
                    markeredgecolor=st['c'],
                    markerfacecolor=st['c'])

        ax.set_title(f'ROC Curve (SNR = {snr_label})', fontweight='bold', fontsize=16)
        ax.set_xlabel('Probability of false alarm', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability of detection', fontsize=14, fontweight='bold')
        
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([0.0, 1.02])
        
        ax.legend(loc='lower right', fontsize=10, framealpha=0.9, fancybox=True)

    plt.tight_layout()
    plt.savefig('roc_with_auc.png', dpi=300)

if __name__ == "__main__":
    run_plot()