import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(
    cm,
    class_names,
    save_path=None,
    normalize=False,
    title='Confusion Matrix',
    cmap='Blues'
):
    """
    Plot confusion matrix
    
    Args:
        cm: confusion matrix array
        class_names: list of class names
        save_path: path to save figure
        normalize: whether to normalize
        title: plot title
        cmap: color map
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set ticks
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate x labels
    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor"
    )
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12
            )
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return fig

def plot_gender_comparison_confusion_matrices(
    cm_male,
    cm_female,
    class_names,
    save_path=None
):
    """Plot side-by-side confusion matrices for gender comparison"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Male confusion matrix
    sns.heatmap(
        cm_male,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={'label': 'Count'}
    )
    axes[0].set_title('Male Patients', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Female confusion matrix
    sns.heatmap(
        cm_female,
        annot=True,
        fmt='d',
        cmap='Reds',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar_kws={'label': 'Count'}
    )
    axes[1].set_title('Female Patients', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gender comparison confusion matrices saved to {save_path}")
    
    plt.show()
    
    return fig