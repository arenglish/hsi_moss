import os
from pathlib import Path
import sys
import glob

sys.path.append(Path(os.getcwd()).as_posix())

from src.hsi_moss.raster import *
from src.hsi_moss.dataset import *
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.svm import SVR
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sklearn import tree
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn import metrics
import pickle

LOAD_EXISTING_MODEL_OPTIMIZATIONS = False

pairpaths = sorted(glob.glob("scripts/svm/modelpair*.pk"))

fig = plt.figure(layout="constrained", dpi=300, figsize=(10, 16))
fig.suptitle("Moss Bioindicator SVR Prediction Models (history: BON)")
gs = gridspec.GridSpec(4, len(pairpaths), figure=fig)

axisunits = {
    "P. Max": "",
    "P. Biom": "(g $cm^{-2}$)",
    "P. Area": "($mm^2$ $cm^{-2}$)",
    "Nitrogen": "",
}

for idx, pairpath in enumerate(pairpaths):
    with open(pairpath, "rb") as f:
        pair = pickle.load(f)
        dx = pair[0]
        dx_name = dx[0]
        dx = dx[1]

        for idy, dy in enumerate(pair[1]):
            dy_name = dy[0]
            dy = dy[1]
            dy.columns = range(dy.shape[1])
            X = dx.loc[:, 2:].values.astype(np.float64)
            y = dy.loc[:, 2].values

            # split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

            # scale data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train[:, np.newaxis])[:, 0]

            # optimize kernel parameters
            C_range = np.exp2(np.array(range(-2, 22, 2)))
            gamma_range = np.exp2(np.array(range(-18, 5, 2)))
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(
                SVR(kernel="rbf"), param_grid=param_grid, cv=cv, verbose=3, n_jobs=-1
            )
            best_params_file = Path(f"scripts/svm/best-params_{dx_name}-{dy_name}.csv")
            if best_params_file.exists() and LOAD_EXISTING_MODEL_OPTIMIZATIONS is True:
                best_params = np.loadtxt(
                    best_params_file.as_posix(), skiprows=1, delimiter=","
                )
            else:
                grid.fit(X_train, y_train.ravel())

                print(
                    "The best parameters are %s with a score of %0.2f"
                    % (grid.best_params_, grid.best_score_)
                )
                best_params = np.array(
                    [grid.best_params_["C"], grid.best_params_["gamma"]]
                )
                np.savetxt(
                    best_params_file.as_posix(),
                    [best_params],
                    header="C,gamma",
                    delimiter=",",
                )

            clf = make_pipeline(
                SVR(
                    kernel="rbf",
                    C=best_params[0],
                    gamma=best_params[1],
                )
            )
            clf.fit(X_train, y_train)

            pr = clf.predict(X_test)[:, np.newaxis]
            pr = scaler_y.inverse_transform(pr)

            ax = fig.add_subplot(gs[idy, idx])
            ax.grid(alpha=0.5)
            ax.set_xlabel(f"Lab {dy_name} " + axisunits[dy_name])
            if idx != 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f"Estimated {dy_name} " + axisunits[dy_name])

            ax.scatter(y_test, pr, color="black", alpha=0.6, s=3, lw=-1)
            # calculate equation for trendline
            z = np.polyfit(y_test, pr, 1)
            p = np.poly1d(z[:, 0])
            r2 = metrics.r2_score(y_test, np.squeeze(pr))
            pearson = pearsonr(np.squeeze(y_test), np.squeeze(pr))
            ax.text(
                0.95,
                0.01,
                "$r^2$ = "
                + format(r2, ".2f")
                + ", d = "
                + format(pearson.correlation, ".2f"),
                verticalalignment="bottom",
                horizontalalignment="right",
                transform=ax.transAxes,
                color="black",
                fontsize=8,
            )
            # add trendline to plot
            min = np.amin([np.amin(y_test), np.amin(pr)])
            max = np.amax([np.amax(y_test), np.amax(pr)])
            ran = max - min
            ax.plot(
                y_test,
                p(y_test),
                color="black",
                alpha=0.3,
                label="$SVR_{" + dx_name + "}$",
            )
            xlims = [min - ran / 10, max + ran / 10]
            ylims = [min - ran / 10, max + ran / 10]
            ax.plot(
                [xlims[0], xlims[1]],
                [ylims[0], ylims[1]],
                color="black",
                lw=0.3,
            )
            ax.set_xlim(xlims[0], xlims[1])
            ax.margins(15)
            ax.tick_params(axis="y", direction="in")
            ax.tick_params(axis="x", direction="in")
            ax.set_ylim(ylims[0], ylims[1])
            ax.legend()
            # plt.show()
fig.axes[3].annotate(
    "$r^{2}$: coefficient of determination\nd: index of agreement",
    xy=(0, -0.3),
    xycoords="axes fraction",
)
fig.savefig("scripts/svm/results.png")
fig.savefig("scripts/svm/results.svg")
fig.savefig("scripts/svm/results.pdf")

print("done")
