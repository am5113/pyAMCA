import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
import matplotlib.ticker as ticker

from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable


from mpl_toolkits.axes_grid1 import make_axes_locatable

def my_heatmap(cm, ax, cmap=plt.cm.Blues):
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=4.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)
    
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] < 2 else "white" )
            
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


def plot_confusion_matrix(y_true, y_pred, classes, ax,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)
    
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    accuracy = 100 * np.mean(y_true==y_pred)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix (Acc: {:.2f}%)'.format(accuracy),
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    ax.grid(False)

def plot_melt_curves(df, NMETA, ylim, filename):
    fig, ax = plt.subplots(1, 9, figsize=(13, 2), dpi=300, constrained_layout=True)
#     fig.suptitle("Melting Curves for MCR targets", fontsize=20, weight="bold")
    ax = ax.flatten()

    df_melt_curve_no_ntc = df.loc[(df.Target!="ntc") & (df.PrimerMix=="PM9.8")]

    for i, (target, df) in enumerate(df_melt_curve_no_ntc.groupby('Target')):
        ax[i].set_title(f"{target}", fontsize=16, weight='bold', c=f"C{i}")
        curves = df.iloc[:, NMETA:].transpose()

        # down-sample number of curves to show
#         curves = curves.sample(200, axis='columns')

        ax[i].plot( curves.index.astype(float), curves.values, c=f"C{i}")
        ax[i].grid(alpha=0.5)
        ax[i].set_ylim(ylim)
        if i>0:
            ax[i].set_yticklabels([])
            
        ax[i].tick_params(axis='x', labelsize=14)
        ax[i].tick_params(axis='y', labelsize=14) 

    plt.savefig(filename)
    plt.show()


def plot_standard_curves(df, NMETA, filename):
    df_ = df.iloc[:, NMETA:].transpose().copy()
    cts = compute_cts(df_, thresh=0.01)

    df_ampl_temp = df.copy()
    df_ampl_temp['CT'] = cts.values

    encod = LabelEncoder()
    df_ampl_temp['Target_enc'] = encod.fit_transform(df_ampl_temp['Target'])

    df_ampl_temp_av = df_ampl_temp.groupby(['Target', 'Conc']).mean().reset_index()

    df_ampl_temp_av['Conc.'] = df_ampl_temp_av['Conc'].apply(lambda x: f'1E{int(np.log10(x))}')

    fig, ax = plt.subplots(1, 1, figsize=(10, 3.5), dpi=300, constrained_layout=True)
    sns.scatterplot(ax=ax, x='Target_enc', y='CT', 
                    data=df_ampl_temp_av, style='Conc.', s=150,  zorder=100,
                    hue='Target_enc', palette=[f'C{i}' for i in range(9)])

    ax.set_xlim((-1, 9.3))
    ax.set_xticks(np.arange(9))
    ax.set_xticklabels(encod.classes_)
    ax.set_xlabel('')

    ax.set_ylabel('')
    ax.set_ylim((0, 40))
    ax.set_yticks(np.arange(0, 40, step=5))

    ax.grid(alpha=0.5, axis='y', linestyle='--')
    ax.grid(which='minor', axis='x')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-9:], labels[-9:], loc='center right')

    
    ax.xaxis.set_minor_locator(ticker.IndexLocator(offset=-0.5, base=1))

    ax.set_xticks([])

    plt.savefig(filename)
    plt.show()

from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
def plot_melt_peak_distributions(df, filename):
    COLORFUL = True

    bins = np.histogram(df['MeltPeaks'], bins=100)[1]
    
    output = pd.DataFrame()

    fig, ax = plt.subplots(1, 1, figsize=(13, 2), dpi=300, constrained_layout=True)
    ax.set_title("qPCR: Melting Curves Distributions per Targets", fontsize=20, weight="bold")

    for i, (target, df_) in enumerate(df.groupby('Target')):  
        if COLORFUL:
            color = f'C{i}'
        else:
            color = 'black'

        g = sns.distplot(df_['MeltPeaks'], bins=bins, color=color, label=target)

        ax.grid(alpha=0.5)
        ax.set_xlim((81.8, 90.5))
        ax.set_xticks(np.arange(82, 90.5, 1.0))

        ax.set_ylabel("PDF", fontsize=16, weight="bold")
        ax.set_xlabel("Melting Temperature (" + chr(176)+ "C)", fontsize=16, weight="bold")

        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14) 

        kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(df_['MeltPeaks'].values.reshape(-1, 1)) 
        x = np.linspace(82, 90, 100)[:, np.newaxis]
        log_density_values = kde.score_samples(x)
        density = np.exp(log_density_values)

        x, y = g.get_lines()[-1].get_data()

        if i==8:
            ax.text(df_['MeltPeaks'].quantile(q=0.2), np.max(y)+1.0, target, color=color, fontsize=12, weight='bold',
                    horizontalalignment='center', verticalalignment='center')
        elif i==1:
            ax.text(df_['MeltPeaks'].quantile(q=0.5), np.max(y)+1.2, target, color=color, fontsize=12, weight='bold',
                    horizontalalignment='center', verticalalignment='center')
            
        else:
            ax.text(df_['MeltPeaks'].quantile(), np.max(y)+0.8, target, color=color, fontsize=12, weight='bold',
                    horizontalalignment='center', verticalalignment='center') 

        ax.set_ylim((0, 6.5))    

        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')

        output[target] = [np.mean(df_['MeltPeaks']), 
                          np.median(df_['MeltPeaks']), 
                          x[np.argmax(y)], 
                          np.std(df_['MeltPeaks']),
                          iqr(df_['MeltPeaks'])]
        
    # ax.legend(ncol=2, fontsize=13, borderpad=0.1, handletextpad=0.1)
    plt.savefig(filename)
    plt.show()
    
    output.index = ['Mean', 'Median', 'MaxLikelihood', 'Std', 'IQR']
    display(output)

def plot_amplification_curves(df, NMETA, ylim, filename):
    fig, ax = plt.subplots(1, 9, figsize=(13, 2), dpi=300, constrained_layout=True)
    # fig.suptitle("qPCR: Amplification Curves per targets", fontsize=20, weight='bold')
    ax = ax.flatten()

    for i, (target, df) in enumerate(df.groupby('Target')):
        ax[i].set_title(f"{target}", fontsize=16, weight='bold', c=f"C{i}")
        curves = df.iloc[:, NMETA:].transpose()
        ax[i].plot( curves.index.astype(float), curves.values, c=f"C{i}")
        ax[i].grid(alpha=0.5)
        ax[i].set_ylim(ylim)

        ax[i].tick_params(axis='x', labelsize=14)
        ax[i].tick_params(axis='y', labelsize=14) 

        if i>0:
            ax[i].set_yticklabels([])

#     fig.text(0.5, -0.05, 'Cycle', ha='center', fontsize=16, weight="bold")
#     fig.text(-0.02, 0.5, 'Amplitude', va='center', rotation='vertical', fontsize=16, weight="bold")

    plt.savefig(filename)
    plt.show()
    
def compute_cts(df, thresh):
    """
    Function to compute Ct for amplification curves.
        df (pd.DataFrame) - dataframe where each column is an amplification curve
        thresh (float) - threshold for computing Ct
    """
    n_rows, n_columns = df.shape

    # Extract insides of df, so as to work with numpy only
    # Note: This is because we will make this function numpy compatible in the future
    cols = df.columns
    x = df.index
    df = df.values

    cts = np.zeros(n_columns)

    for i in range(n_columns):
        y = df[:, i]
        idx, = np.where(y > thresh)
        if len(idx) > 0:
            idx = idx[0]
            p1 = y[idx-1]
            p2 = y[idx]
            t1 = x[idx-1]
            t2 = x[idx]
            cts[i] = t1 + (thresh - p1)*(t2 - t1)/(p2 - p1)
        else:
            cts[i] = -99

    return pd.DataFrame({'Ct': cts}, index=cols)