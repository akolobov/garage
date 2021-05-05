import matplotlib
matplotlib.use('Agg')  # in order to be able to save figure through ssh
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt
from matplotlib import cm
import csv, os, argparse
import numpy as np
# from utils.plot_configs import Configs


from matplotlib import cm
from itertools import chain

SET2COLORS = cm.get_cmap('Set2').colors
SET2 = {'darkgreen': SET2COLORS[0],
        'orange': SET2COLORS[1],
        'blue': SET2COLORS[2],
        'pink': SET2COLORS[3],
        'lightgreen': SET2COLORS[4],
        'gold': SET2COLORS[5],
        'brown': SET2COLORS[6],
        'grey': SET2COLORS[7],
        }

icml_piccolo_final_configs = {
    'model-free': ('Base Algorithm',  SET2['grey']),
    'last': (r'\textsc{Last}', SET2['blue']),
    'replay': (r'\textsc{Replay}', SET2['pink']),
    'sim': (r'\textsc{TrueDyn}', SET2['lightgreen']),
    'sim0.5-VI': (r'\textsc{BiasedDyn0.5-vi}', SET2['orange']),
    'sim0.8-VI': (r'\textsc{BiasedDyn0.8-vi}', SET2['pink']),
    'last-VI': (r'\textsc{Last-vi}', SET2['orange']),
    'sim0.2-VI': (r'\textsc{BiasedDyn0.2-vi}', SET2['darkgreen']),
    'pcl-adv': (r'\textsc{PicCoLO-Adversarial}', SET2['blue']),
    'dyna-adv': (r'\textsc{DYNA-Adversarial}', SET2['pink']),
    'order': [
        'model-free', 'last', 'replay', 'sim', 'sim0.2-VI', 'sim0.5-VI', 'sim0.8-VI', 'last-VI',
        'sim0.2-VI', 'pcl-adv', 'dyna-adv']
}


class Configs(object):
    def __init__(self, style=None, colormap=None):
        if not style:
            self.configs = None
            if colormap is None:
                c1 = iter(cm.get_cmap('Set1').colors)
                c2 = iter(cm.get_cmap('Set2').colors)
                c3 = iter(cm.get_cmap('Set3').colors)
                self.colors = chain(c1, c2, c3)
            else:
                self.colors = iter(cm.get_cmap(colormap).colors)
        else:
            self.configs = globals()[style + '_configs']
            for exp_name in self.configs['order']:
                assert exp_name in self.configs, 'Unknown exp: {}'.format(exp_name)

    def color(self, exp_name):
        if self.configs is None:
            color = next(self.colors)
        else:
            color = self.configs[exp_name][1]
        return color

    def label(self, exp_name):
        if self.configs is None:
            return exp_name
        return self.configs[exp_name][0]

    def sort_dirs(self, dirs):
        if self.configs is None:
            return dirs

        def custom_key(exp_name):
            if exp_name in self.configs['order']:
                return self.configs['order'].index(exp_name)
            else:
                return 100
        return sorted(dirs, key=custom_key)




def configure_plot(fontsize, usetex):
    fontsize = fontsize
    matplotlib.rc("text", usetex=usetex)
    matplotlib.rcParams['axes.linewidth'] = 0.1
    matplotlib.rc("font", family="Times New Roman")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "Times"
    matplotlib.rcParams["figure.figsize"] = 10, 8
    matplotlib.rc("xtick", labelsize=fontsize)
    matplotlib.rc("ytick", labelsize=fontsize)


def truncate_to_same_len(arrs):
    min_len = np.min([x.size for x in arrs])
    arrs_truncated = [x[:min_len] for x in arrs]
    return arrs_truncated


def read_attr(csv_path, attr):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        try:
            row = next(reader)
        except Exception:
            return Nonen
        if attr not in row:
            return None
        idx = row.index(attr)  # the column number for this attribute
        vals = []
        for row in reader:
            vals.append(row[idx])

    vals = [np.nan if v=='' else v for v in vals]
    return np.array(vals, dtype=np.float64)


def main(*,
         logdir, # root directory
         value, #  value to plot
         output_dir=None,
         filename=None,
         style=None,
         y_higher=None,
         y_lower=None,
         n_iters=None,
         legend_loc=0,
         take_max=False,
         curve_style='percentile'):

    log_file_name = 'progress.csv'

    # plot configuration
    fontsize = 32 if style else 12  # for non style plots, exp name can be quite long
    usetex = True if style else False
    configure_plot(fontsize=fontsize, usetex=usetex)
    linewidth = 4
    n_curves = 0

    conf = Configs(style)
    subdirs = sorted(os.listdir(logdir))
    subdirs = [d for d in subdirs if d[0] != '.']  # filter out weird things, e.g. .DS_Store
    subdirs = conf.sort_dirs(subdirs)
    for exp_name in subdirs:
        exp_dir = os.path.join(logdir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

         # load the cvs files from different seeds
        data = []
        for root, _, files in os.walk(exp_dir):
            if log_file_name in files:
                d = read_attr(os.path.join(root, log_file_name), value)
                if d is not None:
                    data.append(d)
        if data:

            n_curves += 1
            data = np.array(truncate_to_same_len(data))

            if take_max:
                data = np.maximum.accumulate(data)

            if curve_style == 'std':
                mean, std = np.mean(data, axis=0), np.std(data, axis=0)
                low, mid, high = mean - std, mean, mean + std
            elif curve_style == 'percentile':
                low, mid, high = np.percentile(data, [25, 50, 75], axis=0)
            if n_iters is not None:
                mid, high, low = mid[:n_iters], high[:n_iters], low[:n_iters]
            iters = np.arange(mid.size)
            # plot
            color = conf.color(exp_name)
            mask =  np.isfinite(mid)
            plt.plot(iters[mask], mid[mask], label=conf.label(exp_name), color=color, linewidth=linewidth)
            plt.fill_between(iters[mask], low[mask], high[mask], alpha=0.25, facecolor=color)

    if n_curves == 0:
        print('Nothing to plot.')
        return 0

    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(value, fontsize=fontsize)
    legend = plt.legend(loc=legend_loc, fontsize=fontsize, frameon=False)
    plt.autoscale(enable=True, tight=True)
    plt.tight_layout()
    plt.ylim(y_lower, y_higher)
    plt.grid(linestyle='--', linewidth='0.2')
    for line in legend.get_lines():
        line.set_linewidth(6.0)
    output_dir = output_dir or logdir
    output_filename = filename or '{}.pdf'.format(str.replace(value,'/','-'))
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--logdir', help='The dir of experiments', type=str)
    parser.add_argument('-v', '--value', help='The column name in the log.txt file', type=str, default='Evaluation/AverageReturn')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-f,','--filename', type=str, default='', help='Output filename')
    parser.add_argument('--style', type=str, default='', help='Plotting style')
    parser.add_argument('--y_higher', nargs='?', type=float)
    parser.add_argument('--y_lower', nargs='?', type=float)
    parser.add_argument('--n_iters', nargs='?', type=int)
    parser.add_argument('--legend_loc', type=int, default=0)
    parser.add_argument('--curve_style', type=str, default='percentile', help='percentile, std')

    args = parser.parse_args()

    main(**vars(args))
