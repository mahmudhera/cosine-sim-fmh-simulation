import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "tahoma"
plt.rcParams['font.size'] = 11

if __name__ == "__main__":
    df = pd.read_csv('cosine_simulation_results')
    fig, axs = plt.subplots(2, 2, sharey=True)
    a, b = min(df['cos_org'].tolist()), max(df['cos_org'].tolist())

    for s, coord in zip([0.0001, 0.001, 0.01, 0.1], [ [0,0], [0,1], [1,0], [1,1] ]):
        df2 = df[ (df['s'] == s) ]
        axs[coord[0],coord[1]].scatter(df2['cos_org'], df2['cos_fmh'], s=1, color='#d896ff')
        axs[coord[0],coord[1]].plot( [a, b], [a,b], color='#800080')
        axs[coord[0],coord[1]].set_title(f'FMH Scale Factor {s}', fontsize=11)

    for ax in axs.flat:
        ax.set(xlabel='True cosine', ylabel='Cosine  using FMH')
    fig.suptitle('Estimated cosine similarity for various scale factors', fontsize=11)
    plt.subplots_adjust(hspace=0.5)

    plt.savefig('true_vs_fmh_cosine.pdf')
