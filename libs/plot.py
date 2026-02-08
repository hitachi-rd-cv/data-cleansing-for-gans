import matplotlib as mpl
import numpy as np
import seaborn as sns
from PIL import Image
from matplotlib import cm, pyplot as plt
from scipy import stats
from sklearn import metrics

from libs.image import get_3ch_scaled_imgs, get_tiles_of_array
from libs.misc import load
from scipy.special import comb

def expected_indepencent_jaccard(nsamples, j_size):
    # thank you joriki! https://math.stackexchange.com/a/1770628
    n = nsamples
    m = j_size
    expected_jac = 0
    for k in np.arange(m + 1):
        expected_jac += k / (2 * m - k) * comb(m, k) * comb(n - m, m - k) / comb(n, m)
    return expected_jac


def gen_plot_fig_with_colored_dots_and_legends(x, y, labels, path, _run=None):
    assert len(x) == len(y) == len(labels)
    labels = np.array(labels)
    for label in np.unique(labels):
        is_label = labels == label
        plt.plot(x[is_label], y[is_label], '.', label=label)
    plt.legend(loc='upper right', fontsize=8)
    plt.tick_params(labelsize=8)
    plt.savefig(path)

    if _run is not None:
        _run.add_artifact(path)


def gen_error_fig(
        actual, approx, _run=None, zero_centered=True,
        title='Errors between actual and approximated',
        xlabel='Actual',
        ylabel='Approximated',
        scores='r2',
        jaccard_size=0.1,
):
    assert len(actual) == len(approx), f'{len(actual)} != {len(approx)}'
    if zero_centered:
        max_abs = np.max([np.abs(actual), np.abs(approx)])
        min_, max_ = -max_abs * 1.1, max_abs * 1.1
        pos_text = [max_abs, -max_abs]
    else:
        min_ = np.min([actual, approx])
        max_ = np.max([actual, approx])
        m = np.abs(max_ - min_) * 0.1
        min_ -= m
        max_ += m
        pos_text = [max_, min_]

    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.0E"))
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
    ax.scatter(actual, approx, zorder=2, s=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    range_ = [min_, max_]
    ax.plot(range_, range_, 'k-', alpha=0.2, zorder=1)
    texts = ['# of samples = {}'.format(len(actual))]
    d_score = {}
    for score in scores:
        if score == 'r2':
            val = metrics.r2_score(actual, approx)
            text = 'R2 score = {:.03}'.format(val)
            d_score[score] = val
        elif score == 'mae':
            val = metrics.mean_absolute_error(actual, approx)
            text = 'MAE = {:.03}'.format(val)
            d_score[score] = val
        elif score == 'kendalltau':
            val = stats.kendalltau(actual, approx)[0]
            text = 'Kendall tau = {:.03}'.format(val)
            d_score[score] = val
        elif score == 'jaccardindex':
            n_samples_jaccard = int(len(actual) * jaccard_size / 2)
            base_indices = np.arange(len(actual))
            sorted_indices_actual = np.argsort(actual)
            influential_indices_actual = np.concatenate([sorted_indices_actual[:n_samples_jaccard], sorted_indices_actual[-n_samples_jaccard:]])
            is_influenctial_actual = np.isin(base_indices, influential_indices_actual)
            sorted_indices_approx = np.argsort(approx)
            influential_indices_approx = np.concatenate([sorted_indices_approx[:n_samples_jaccard], sorted_indices_approx[-n_samples_jaccard:]])
            is_influenctial_approx = np.isin(base_indices, influential_indices_approx)
            pred_score = metrics.jaccard_score(is_influenctial_actual, is_influenctial_approx)
            expected_score = expected_indepencent_jaccard(len(actual), n_samples_jaccard*2)
            text = 'Jaccard index ({}/{}) = {:.03} ({:.03})'.format(n_samples_jaccard*2, len(actual), pred_score, expected_score)
            d_score[score] = pred_score
        else:
            raise ValueError(score)
        texts.append(text)

    text_cat = '\n'.join(texts)
    ax.text(*pos_text, text_cat, fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)
    fig.subplots_adjust(bottom=0.2)

    return fig, ax, d_score


def plot_filling_std(ax, xs, plots, xlabel=None, ylabel=None, log_scale=False, loc_legend='lower right', keys_ignored_by_autoscale=[], fill=True, _run=None):
    keys = {k: v for k, v in plots.items() if k not in keys_ignored_by_autoscale}
    for k in keys:
        eval_metric_vals = np.asarray(plots[k])
        means, stds = np.mean(eval_metric_vals, axis=0), np.std(eval_metric_vals, axis=0)
        ax.plot(xs, means, label=k, marker='o')
        if _run is not None:
            for mean in means:
                _run.log_scalar(k, mean)
        if fill:
            ax.fill_between(xs, means - stds, means + stds, alpha=.1)

    keys_ignored_by_autoscale = {k: v for k, v in plots.items() if k in keys_ignored_by_autoscale}
    ylims = ax.get_ylim()
    for k in keys_ignored_by_autoscale:
        eval_metric_vals = np.asarray(plots[k])
        means, stds = np.mean(eval_metric_vals, axis=0), np.std(eval_metric_vals, axis=0)
        ax.plot(xs, means, label=k, marker='o')
        if _run is not None:
            for mean in means:
                _run.log_scalar(k, mean)
        if fill:
            ax.fill_between(xs, means - stds, means + stds, alpha=.1)
    ax.set_ylim(ylims)

    if log_scale:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if loc_legend == 'outside':
        ax.legend(loc="lower left", bbox_to_anchor=(1.02, 0.0,), borderaxespad=0)
    else:
        ax.legend(loc=loc_legend)
    return ax


def jointplot(path, *args, _run=None, **kwargs):
    grid = sns.jointplot(*args, **kwargs)
    # grid.plot_joint(kde_density_ratio_plot, hue_name='Generated (before cleansing)', cut=0)
    grid.savefig(path)

    if _run is not None:
        _run.add_artifact(path)


def get_plot(path, val, names=None, color=None, figsize=(5, 5), mode='scatter', _run=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(val, (list, tuple)):
        if names is None:
            names = [None] * len(val)
        if color is None:
            color = [None] * len(val)
        assert len(val) == len(color)
        for c, x, n in zip(color, val, names):
            if mode == 'kde':
                kde(ax, x, c, label=n, **kwargs)
                # ax.scatter(*x.mean(axis=0), marker='+', color=c)
            elif mode == 'scatter':
                scatter(ax, x, c, label=n, **kwargs)
            elif mode == 'contour':
                ax.contour(*x.T, label=n)
            else:
                raise ValueError(mode)
    else:
        if mode == 'kde':
            kde(ax, val, color)
        elif mode == 'scatter':
            scatter(ax, val, color)
        elif mode == 'contour':
            import matplotlib.tri as tri
            ngridx = 100
            ngridy = 200
            npts = 200

            x, y, z = val.T

            # Create grid values first.
            xi = np.linspace(x.min() - 1, x.max() + 1, ngridx)
            yi = np.linspace(y.min() - 1, y.max() + 1, ngridy)

            # Perform linear interpolation of the data (x,y)
            # on a grid defined by (xi,yi)
            triang = tri.Triangulation(x, y)
            interpolator = tri.LinearTriInterpolator(triang, z)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            # Note that scipy.interpolate provides means to interpolate data on a grid
            # as well. The following would be an alternative to the four lines above:
            # from scipy.interpolate import griddata
            # zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')

            ax.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
            cntr1 = ax.contourf(xi, yi, zi, levels=14, cmap="RdBu_r")

            fig.colorbar(cntr1, ax=ax)
            # ax.plot(x, y, 'ko', ms=3)
            # ax.set(xlim=(-2, 2), ylim=(-2, 2))
            ax.set_title('grid and contour (%d points, %d grid points)' %
                         (npts, ngridx * ngridy))

        else:
            raise ValueError(mode)

            # ax.contour(*val.T)

    ax.axis('off')
    fig.savefig(path)

    if _run is not None:
        _run.add_artifact(path)


def kde(ax, x, c, n_levels=20, **kwargs):
    return sns.jointplot(x=x[:, 0], y=x[:, 1], cmap=c, kind="kde", ax=ax, n_levels=n_levels, **kwargs)


def scatter(ax, x, c, **kwargs):
    return ax.scatter(x[:, 0], x[:, 1], c=c, edgecolor='none', **kwargs)



def merge_line_legends(figs, line_names, indices_sort, ncol=1):
    legends = []
    for fig in figs:
        for line, line_name in zip(*fig.axes[0].get_legend_handles_labels()):
            legends.append(line)
    sorted_legends = [legends[i] for i in indices_sort]

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(sorted_legends, line_names, frameon=False, loc='lower center', ncol=ncol)
    fig_legend = legend.figure
    fig_legend.canvas.draw()
    return legend

def get_tight_legend_bbox_inches(legend):
    bbox = legend.get_window_extent().transformed(legend.figure.dpi_scale_trans.inverted())
    return bbox

def get_colorbar_figure(figsize, rect, cmap='inferno_r'):
    # Make a figure and axes with dimensions as desired.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    # Set the colormap and norm to correspond to the data for which
    # the colorbar will be used.
    cmap = cm.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=50)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Ranking of predicted harmful score [k th] \n(lower are more harmful)')
    fig.tight_layout()

    return fig, ax


def plot_image_table(nested_images, row_labels, col_labels, figsize=(20, 20)):
    assert len(nested_images) == len(row_labels)
    assert all([len(images) == len(col_labels) for images in nested_images])

    fig, axes = plt.subplots(len(row_labels), len(col_labels), figsize=figsize)
    for i, (images, row_label) in enumerate(zip(nested_images, row_labels)):
        for j, (image, col_label) in enumerate(zip(images, col_labels)):
            ax = axes[i][j]
            ax.set_xlabel(col_label)
            ax.set_ylabel(row_label)
            ax.imshow(Image.fromarray(image.astype(np.uint8)))
            ax.set_xticks([])
            ax.set_yticks([])


def get_img_tiles_by_row_vals(row_vals, col_names, callback_img_path, callback_replace_xs, nfirst_imgs=36):
    x_tiles_by_metrics = []
    for i, row_val in enumerate(row_vals):
        x_tiles = []
        for name in col_names:
            path = callback_img_path(row_val, name)
            xs = load(path)
            xs_scaled = get_3ch_scaled_imgs(xs[:nfirst_imgs], 255. / 2., 1.)
            xs_scaled = callback_replace_xs(row_val, name, xs_scaled)
            x_tile = get_tiles_of_array(xs_scaled)
            x_tiles.append(x_tile)
        x_tiles_by_metrics.append(x_tiles)

    return x_tiles_by_metrics


# plot training instances with scores given by each selection method
def scatter_with_score_rank_color(xs, scores, xlabel=None, ylabel=None, figsize=(4, 4), s=2, cmap='inferno_r'):
    normalized_scores = (-scores).argsort().argsort()
    fig, ax = plt.subplots(figsize=figsize)
    scatter(ax, xs, normalized_scores, s=s, cmap=cm.get_cmap(cmap))
    fig.tight_layout()

    return fig, ax


def _quantile_to_level(data, quantile):
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels


def plot_true_2dnormal(ax, x, y, data, mean, cov, nlevels_contour=8, idx_color=0):
    sns.distributions._DistributionPlotter(data=data)
    kde = sns._statistics.KDE()
    support = kde.define_support(data[x], data[y])
    xx1, xx2 = np.meshgrid(*support)
    pos = np.dstack((xx1, xx2))
    mv = stats.multivariate_normal(mean, cov)
    density = mv.pdf(pos)
    xx, yy = support
    levels = np.power(np.arange(nlevels_contour) / nlevels_contour, 2)
    draw_levels = _quantile_to_level(density, levels)
    color = sns.color_palette()[idx_color]
    ax.contour(
            xx, yy, density,
            levels=draw_levels,
            colors=[color]
    )
    return ax


def plot_kde_and_true_2dnormal(x, y, data, hue, nlevels_contour, figsize, mean, cov, idx_color_true, idxs_color_other):
    fig, ax = plt.subplots(figsize=figsize)
    ax = plot_true_2dnormal(ax, x, y, data, mean, cov, nlevels_contour, idx_color_true)
    ax = sns.kdeplot(x=x,
                     y=y,
                     data=data,
                     hue=hue,
                     levels=np.power(np.arange(nlevels_contour) / nlevels_contour, 2),
                     legend=False,
                     ax=ax,
                     common_norm=False,
                     palette=[sns.color_palette()[i] for i in idxs_color_other])
    fig.tight_layout()

    return fig, ax


def plot_colored_hist(xs_true, approx_diffs, nbins=10, inverse_colormap=False, name_cmap="bwr"):
    # Define the colormap
    colormap = plt.get_cmap(f'{name_cmap}{"_r" if inverse_colormap else ""}')

    # Calculate histogram
    hist, bin_edges = np.histogram(xs_true, bins=nbins)

    # Get the bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate mean approx_diffs for each bin
    indices = np.digitize(xs_true, bin_edges)
    mean_diffs = np.array([np.mean(approx_diffs[indices == i]) for i in range(1, len(bin_edges))])

    # Normalize mean_diffs for color mapping
    # Note: we use 75 percentile of the absolute maximum value to ensure that 0 maps to the center color of the colormap.
    max_abs_val_quantile = np.percentile(np.abs(mean_diffs), 75)
    norm = plt.Normalize(-max_abs_val_quantile, max_abs_val_quantile)

    # Create color array for barplot
    colors = colormap(norm(mean_diffs))

    # Create the histogram
    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist, width=np.diff(bin_edges), color=colors, align='edge', edgecolor='black')

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm)

    return fig, ax


def get_legend(labels, markers, colors, styles, markersizes=None, ncol=1):
    if markersizes is None:
        markersizes = [None] * len(markers)
    assert len(labels) == len(markers) == len(colors)

    fig, ax = plt.subplots()
    for label, marker, color, style, markersize in zip(labels, markers, colors, styles, markersizes):
        ax.plot(np.ones(1), np.ones(1), label=label, marker=marker, color=color, linestyle=style, markersize=markersize)
    legends = []
    for line, line_name in zip(*fig.axes[0].get_legend_handles_labels()):
        legends.append(line)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.axis('off')
    legend = ax2.legend(legends, labels, frameon=False, loc='lower center', ncol=ncol)
    fig_legend = legend.figure
    fig_legend.canvas.draw()

    return legend