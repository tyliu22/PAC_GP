

import matplotlib.pylab as plt
import seaborn as sns

import matplotlib.pylab
import numpy as np
import os
import pandas



matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

labels = {'GPflow Full GP': 'full-GP',
          'bkl-PAC HYP GP': 'kl-PAC-GP',
          'sqrt-PAC HYP GP': 'sqrt-PAC-GP',
          'sqrt-PAC Inducing Hyp NIGP': 'sparse PAC-NIGP',
          'bkl-PAC Inducing Hyp GP': 'kl-PAC-SGP',
          'sqrt-PAC Inducing Hyp GP': 'sqrt-PAC-SGP',
          'GPflow VFE': 'VFE',
          'GPflow FITC': 'FITC'
          }

colors = {'bkl-PAC HYP GP': sns.color_palette("Paired")[0],
          'bkl-PAC Inducing Hyp GP': sns.color_palette("Paired")[0],
          'sqrt-PAC Inducing Hyp GP': sns.color_palette("Paired")[1],
          'sqrt-PAC HYP GP': sns.color_palette("Paired")[1],
          'sqrt-PAC Inducing Hyp NIGP': sns.color_palette("Paired")[1],
          'GPflow Full GP': sns.color_palette("Paired")[4],
          'GPflow FITC': sns.color_palette("Paired")[3],
          'GPflow VFE': sns.color_palette("Paired")[2],
          }


def plot_lines(ax, D, x, metric, models, ylabel=None, xticks=None, ylim=None,
               yticks=None, legend=False):
    """
    beautfiy comparison plot
    """
    bars = []
    for i, model in enumerate(models):
        # D_temp = D[(D['model'] == model) & (D['metric'] == metric)]
        # D_filtered = D_temp[(D_temp['nInd'] == 80)]
        D_filtered = D[(D['model'] == model) & (D['metric'] == metric)]
        if metric == 'KL-divergence':
            D_filtered = D_filtered.reset_index()
            D_filtered['value'] = D_filtered['value'].div(D_filtered.N, axis=0)

        agg_params = ['size', 'mean', 'var']
        D_filtered = D_filtered.groupby(x).agg(agg_params).reset_index()
        mean = D_filtered['value']['mean']
        stderr = np.sqrt(D_filtered['value']['var'])
        stderr /= np.sqrt(D_filtered['value']['size'])

        _x = np.arange(len(mean))
        _bar = plt.bar(_x + 0.2*i, mean, label=labels[model], width=0.20,
                       yerr=stderr, color=colors[model],
                       ecolor='#696969')
        bars.append(_bar)

    plt.xticks(_x + 0.2, D_filtered[x])
    plt.xlabel(x)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)
    if legend:
        plt.legend(bbox_to_anchor=(-5.5, 1.02, 4, 0.1), loc=3, ncol=3,
                   mode="expand", borderaxespad=0., frameon=False)
        # plt.legend(bbox_to_anchor=(-5.5, 1.02, 4, 0.1), loc=3, ncol=3,
                   # mode="expand", borderaxespad=0., frameon=False)

    return bars


def plot_(D, models, x, xticks=None, ylim=None, legend=True, yticks=None):
    """
    plotting results with respect to
    - upper-bound-bkl
    - gibbs-risk-train
    - gibbs-risk
    - MSE
    - KL-divergence
    """
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(left=0.05, bottom=0.2, wspace=0.40, right=1.0)

    ax = fig.add_subplot(151)
    plot_lines(ax, D, x, 'upper-bound-sqrt', models, ylabel='Upper Bound',
               ylim=(0, 0.6), yticks=yticks)

    ax = fig.add_subplot(152)
    plot_lines(ax, D, x, 'KL-divergence', models, ylabel='KL divergence')

    ax = fig.add_subplot(153)
    plot_lines(ax, D, x, 'gibbs-risk-train', models, ylabel='$R_S$[Train]',
               ylim=(0, 0.5), yticks=yticks)

    ax = fig.add_subplot(154)
    plot_lines(ax, D, x, 'gibbs-risk', models, ylabel='$R_S$[Test]',
               ylim=(0, 0.5), yticks=yticks)

    ax = fig.add_subplot(155)
    bars = plot_lines(ax, D, x, 'MSE', models, ylabel='MSE', ylim=(0, 0.6), legend=legend)









if __name__ == '__main__':
    result_dir = 'results_rebuttal'
    # fn_base = 'boston_01_loss_S_ARD0_eps0.6_testsize20_nReps1' % fn_args
    # fn_results = os.path.join('ind_points', '%s.pckl' % fn_base)
    # if not(os.path.exists('ind_points')):
    #     os.mkdir('ind_points')
    fn_results = "/home/tianyliu/Data/remote/PAC_GP/Results/boston_01_loss_S_ARD0_eps0.6_testsize20_nReps1.pckl"
    D = pandas.read_pickle(fn_results)
    models = ['sqrt-PAC Inducing Hyp NIGP',
                  'GPflow VFE', 'GPflow FITC']
    plot_(D, models, x="nInd", xticks=[0.2, 0.4, 0.6, 0.8, 1.0],
                      ylim=(0, 0.85))
    # fn_png = os.path.join('ind_points', '%s.png' % fn_base)
    fn_pdf = "/home/tianyliu/Data/remote/PAC_GP/Results/boston_01_loss_S_ARD0_eps0.6_testsize20_nReps.pdf"
    # plt.savefig(fn_png)
    plt.savefig(fn_pdf)
    plt.show()
    print('End')
