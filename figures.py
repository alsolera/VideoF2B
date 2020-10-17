# VideoF2B - Draw F2B figures from video
# Copyright (C) 2020  Andrey Vasilik - basil96@users.noreply.github.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''Geometric definitions for F2B figures.'''

import itertools
from collections import defaultdict

import matplotlib.pyplot as plt  # for debug diagnostics
import numpy as np
from scipy import optimize

import common

# ==================== Golden Section Search method ===============================
# NOTE: Unused as of yet. Also available in scipy.optimize


def find_min_gss(f, a, b, eps=1e-4):
    '''Find Minimum by Golden Section Search Method
        Returns the value of x that minimizes the function f(x) on interval [a, b]
    '''
    # Golden section: 1/phi = 2/(1+sqrt(5))
    R = 0.61803399
    # Num of needed iterations to get precision eps: log(eps/|b-a|)/log(R)
    n_iter = int(np.ceil(-2.0780869 * np.log(eps / np.abs(b - a))))
    c = b - (b - a) * R
    d = a + (b - a) * R
    for i in range(n_iter):
        if f(c) < f(d):
            b = d
        else:
            a = c
        c = b - (b - a) * R
        d = a + (b - a) * R
    return (b + a) / 2


class FigureDiagnostics:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.args_low = None
        self.args_high = None
        self.errs_hist = None
        self.fit_params = None
        self.u0 = None
        self.trim_indexes = None


class Figure:
    '''Base class for all F2B figures.'''

    def __init__(self, R=None, actuals=None, **kwargs):
        '''Create a figure in a sphere of radius R associated with given actual points.'''

        self.diag = FigureDiagnostics(enabled=kwargs.pop('enable_diags', False))

        if R is None:
            self.R = common.DEFAULT_FLIGHT_RADIUS
        else:
            self.R = R

        self._actuals = actuals
        # Initial pathwise t-parameterization
        self._init_path()

        if self._actuals is not None:
            act_centroid = self._actuals.mean(axis=0)
            # print(f'\n\nact_centroid = {act_centroid}\n\n')
            # Azimuth of the centroid of the actual points (rough estimate)
            self.phi0 = np.arctan2(act_centroid[1], act_centroid[0])
        else:
            self.phi0 = 0.

        '''Suggested count of nominal points for a reasonably smooth-looking figure.
        Override in subclasses as needed. This is primarily used for drawing.'''
        self.num_nominal_pts = 100

        # Reference to each subclassed Figure's parametric function
        self.paramfunc = self._parmfn()

    @staticmethod
    def create(which_figure, R=None, actuals=None, **kwargs):
        '''Factory method for creating a given Figure on a sphere of radius R
        with specified actual path points.'''
        mapper = {
            FigureTypes.INSIDE_LOOPS: InsideLoops,
        }
        fig_type_constructor = mapper.get(which_figure, None)
        if fig_type_constructor:
            return fig_type_constructor(R=R, actuals=actuals, **kwargs)
        else:
            raise NotImplementedError(
                f'*** T O D O ***: Figure type {which_figure} is not implemented.')

    def fit(self):
        '''Perform best fit of actual points against the nominals of the figure.'''

        print(f'shape of [u] on entry to Figure.fit(): {self.u.shape}')

        resids = defaultdict(list)
        fit_idx = 0

        def get_ext_residuals(p, points, u):
            '''Objective function for least-squares fit'''
            # print('in get_ext_residuals')
            nonlocal resids
            nonlocal fit_idx
            # print(f'fit params = {p}')  # applies to all subclasses
            # print(f'fit params = {np.degrees(p[0]):.3f}°, {np.degrees(p[1]):.3f}°, {p[2]:.3f}') # NB: applies only to InsideLoops class
            # fits.append(p)
            # p_u = np.array([self.paramfunc(u, *p) for u in self.u])
            # print(f'p = {p}')
            p_u = self.paramfunc(*p, *u)
            diff = points - p_u
            # print(f'diff: {diff.shape}')
            # print(f'diff =\n{diff[10:20]}')
            dists = np.linalg.norm(diff, axis=1)  # ** 2
            eps = dists
            # print(f'eps = {eps}, p = {p}')
            resids[fit_idx].append(eps)
            return eps

        def get_int_residuals(u, points, p):
            '''Objective function for optimizing the internal t values along the path.'''
            # print('in get_int_residuals')
            nonlocal resids
            nonlocal fit_idx
            p_u = self.paramfunc(*p, *u)
            diff = points - p_u
            dists = np.linalg.norm(diff, axis=1)
            eps = dists
            resids[fit_idx].append(dists)
            return eps

        fit_params = {}
        fit_errs = {}

        c = itertools.cycle('=@#@#@#@#@#@#@#@#@#@#@#@#')
        extrinsics = []
        NUM_FITS = 3
        for fit_idx in range(NUM_FITS):
            print(next(c) * 80)
            if fit_idx % 2 == 0:
                print(f'Starting fit #{fit_idx}: external figure parameters...')
                if fit_idx == 0:
                    x0 = self.p0
                    arg2 = self.u
                else:
                    x0 = fit_params[fit_idx - 2]
                    arg2 = fit_params[fit_idx - 1]
                fit_result = optimize.least_squares(
                    get_ext_residuals, x0,
                    args=(self._actuals, arg2),
                    bounds=([-np.inf, -np.inf, 1.], [np.inf, np.inf, np.inf]),
                    xtol=0.002
                )
                fit_params[fit_idx], fit_errs[fit_idx] = fit_result.x, fit_result.status
                print(f'Fit #{fit_idx} done, results = {fit_params[fit_idx], fit_errs[fit_idx]}\n')
                extrinsics.append(fit_params[fit_idx])
            else:
                if fit_idx == 1:
                    x0 = self.u
                else:
                    x0 = fit_params[fit_idx - 2]
                arg2 = fit_params[fit_idx - 1]
                print(f'Starting fit #{fit_idx}: internal t parameters...')
                fit_result = optimize.least_squares(
                    get_int_residuals, x0, args=(self._actuals, arg2),
                    bounds=(-50., 50.),
                    ftol=0.001
                )
                fit_params[fit_idx], fit_errs[fit_idx] = fit_result.x, fit_result.status
                self.diag.args_low = list(np.argwhere(fit_params[fit_idx] < 0.0).flat)
                self.diag.args_high = list(np.argwhere(fit_params[fit_idx] > 1.0).flat)
                print(f'Fit #{fit_idx} done, results:')
                print(f'shape of u: {self.u.shape}')
                print(
                    f'args where t < 0:\n{self.diag.args_low}\nlen: {len(self.diag.args_low)}')
                print(
                    f'args where t > 1: {self.diag.args_high}\nlen: {len(self.diag.args_high)}')

        # Histories of the fitting journey (for diagnostics only). We save one per each get_*_residuals() call.
        self.diag.errs_hist = resids  # history of residual sets

        if self.diag.enabled:
            resid_sums = {}
            for f_idx, f_resids in resids.items():
                num_iterations = len(f_resids)
                s_resids = [sum(fr) for fr in f_resids]
                resid_sums[f_idx] = s_resids
                print(f'f_idx={f_idx}, num_iterations={num_iterations}')
                if num_iterations == 0:
                    continue
                # Plot history of errors during optimization
                plt.figure(f_idx)
                plt.subplot(1, 2, 1)
                for i, err in enumerate(f_resids):
                    if 0 < i < num_iterations - 1:
                        plt.plot(err, color='tab:gray', marker=None, linestyle='--',
                                 label='Intermediate' if i == 1 else None)
                plt.plot(f_resids[0], 'r.-', label=f'Initial, sum={sum(f_resids[0])}')
                plt.plot(f_resids[-1], 'b.-', label=f'Optimized, sum={sum(f_resids[-1])}')
                plt.title(f'Errors during fit #{f_idx} ({num_iterations} iterations)')
                # Plot trend of error sums during optimization
                plt.subplot(1, 2, 2)
                plt.semilogy(s_resids, ',-')
                plt.title(f'Trend of residual sums during fit #{f_idx}')
            # Plot initial and final pathwise parameter distributions
            plt.figure(f_idx + 1)
            plt.plot(self.u, 'r.-')
            plt.plot(fit_params[f_idx - 1], 'b.-')
            plt.plot((0, len(self.u)), (0., 0.), 'r--')
            plt.plot((0, len(self.u)), (1., 1.), 'r--')
            plt.title('Initial and final $u_i$')
            # Plot the histories of extrinsic parameter sets and their corresponding residual sums
            plt.figure(f_idx + 2)
            # Each row is a 1-d array of extrinsic param values
            extrinsics = np.asarray(extrinsics)
            # Each row is a 1-d array of deltas in the extrinsic param value from the previous
            trends = np.diff(extrinsics, axis=0)
            for i, trend in enumerate(trends):
                # Note: indexes of the extrinsic fits are always even
                f_idx_ref = 2 * i + 2
                p_labels = [f'$p_{i}$' for i in range(len(trend))]
                f_legend = f'$fit_{f_idx_ref}$, sum={resid_sums[f_idx_ref][-1]:.6f}'
                plt.plot(p_labels, trend.T, '.-', label=f_legend)
            plt.legend()
            plt.title('Trends: extrinsic parameters at each iteration')

            plt.show()

        self.diag.fit_params = fit_params
        self.diag.u0 = self.u.copy()
        print(f'shape of [u] before trim: {self.u.shape}')
        candidates = fit_params[1]
        print(f'shape of candidates: {candidates.shape}')
        self.diag.trim_indexes = np.logical_and(candidates >= 0.0, candidates <= 1.0)
        print(f'shape of trim_indexes where 0 < u_i < 1: {self.diag.trim_indexes.shape}')
        self.u = candidates[self.diag.trim_indexes]
        print(f'shape of [u] after trim: {self.u.shape}')
        return fit_params[fit_idx], fit_errs[fit_idx]

    # def _calc_t(self):
    #     # chordal parameterization
    #     norms = np.linalg.norm(np.diff(self._actuals, axis=0), axis=1)
    #     # # centripetal parameterization (doesn't work so well here)
    #     # norms = np.linalg.norm(np.diff(self._actuals, axis=0), axis=1) ** 2
    #     self.u = np.hstack(
    #         (0.0, np.cumsum(norms) / sum(norms))
    #     )
    #     u_diff = np.diff(self.u)
    #     # cum_diff = np.diff(np.cumsum(norms))
    #     # Use 1% as the threshold??
    #     outliers_mask = np.logical_not(np.hstack((False, u_diff > 0.01)))
    #     return outliers_mask

    # def _init_path(self):
    #     '''Creates the initial parameterization: chordal type'''
    #     if self._actuals is not None:
    #         u_outliers_mask = self._calc_t()
    #         if not u_outliers_mask.all():
    #             self._actuals = self._actuals[u_outliers_mask]
    #             self._calc_t()
    #     else:
    #         self.u = np.linspace(0., 1., self.num_nominal_pts)

    #     if self.diag.enabled:
    #         print(f'shape of initial [u]: {self.u.shape}')
    #         plt.figure(100)
    #         # plt.subplot(3, 1, 1)
    #         plt.plot(self.u)
    #         plt.title('Initial parameterization')
    #         # plt.subplot(3, 1, 2)
    #         # plt.plot(u_diff, '.-')
    #         # plt.title('First diff of initial parameterization')
    #         # plt.subplot(3, 1, 3)
    #         # plt.plot(cum_diff, '.-')
    #         # plt.title('First diff of cumulative sum of norms')
    #         plt.show()

    def _init_path(self):
        '''Creates the initial parameterization: chordal type'''
        # chordal parameterization
        norms = np.linalg.norm(np.diff(self._actuals, axis=0), axis=1)
        # # centripetal parameterization (doesn't work so well here)
        # norms = np.linalg.norm(np.diff(self._actuals, axis=0), axis=1) ** 2
        self.u = np.hstack(
            (0.0, np.cumsum(norms) / sum(norms))
        )
        # print(self.u[:5], self.u[-5:], self.u.shape)
        # plt.plot(self.u)
        # plt.title('Initial parameterization')
        # plt.show()

    def get_nom_point(self, a, b, c, *t):
        '''Returns the nominal point at a given 0.0 < t < 1.0 using the figure's parameters.'''
        # TODO: make these function signatures more generic. Currently this supports only 3 params and t.
        return self.paramfunc(a, b, c, *t)

    # def get_nom_points(self, *params, t_arr):
    #     '''Return an array of points, one for each t in t_arr'''
    #     return np.array([self.paramfunc(*params, t) for t in t_arr])


class InsideLoops(Figure):
    '''Represents three consecutive inside loops per F2B Rule 4.2.15.5 and Diagram 4.J.3 in the Annex.
    kwargs:
        `enable_diags` : enables diagnostic output and plotting of various behind-the-scenes stuff.
    '''

    def __init__(self, R=None, actuals=None, **kwargs):
        super().__init__(R=R, actuals=actuals, **kwargs)
        # Nominal point count: every 5 degrees of the loop
        self.num_nominal_pts = 252
        # Initial guesses for optimization params
        self.p0 = [self.phi0, self.theta, self.r]
        # print(f'self.p0 = {self.p0}')
        # self.p0 = [0., 0., 1.] # rather extreme case, just to test sensitivity/convergence

    def fit(self):
        '''Fit actual flight path to this Figure's nominal path.'''

        print('Initial parameters:')
        print(f'  phi   = {self.p0[0]} ({np.degrees(self.p0[0]):.3f}°)')
        print(f'  theta = {self.p0[1]} ({np.degrees(self.p0[1]):.3f}°)')
        print(f'  r     = {self.p0[2]}')

        p_fit, fit_err = super().fit()

        print(f'Optimized parameters: {p_fit, fit_err}')
        print(f'  phi   = {p_fit[0]} ({np.degrees(p_fit[0]):.3f}°)')
        print(f'  theta = {p_fit[1]} ({np.degrees(p_fit[1]):.3f}°)')
        print(f'  r     = {p_fit[2]}')

        return p_fit

    def _parmfn(self):
        '''Parameterizing function for three consecutive inside loops.'''
        #### Values that don't depend on figure parameters. Calculate these once per instance. ####
        # Included angle of cone
        alpha = np.radians(45.0)
        # Elevation angle of loop normal
        self.theta = np.radians(22.5)
        # Parametric angle inside loop when t = 1 (3.5 loops)
        k = 7.0 * np.pi
        # Radius of loop
        self.r = self.R * np.sin(0.5 * alpha)
        print(f'Nominal R = {self.R}')
        # print(f'Nominal phi = {self.phi0} ({np.degrees(self.phi0):.3f}°)')
        # print(f'Nominal theta = {self.theta} ({np.degrees(self.theta):.3f}°)')
        print(f'Nominal r = {self.r}')

        def func(phi, theta, r, *t):
            '''Returns all points on the nominal loops' path according to the specified parameters:
                    `t`: 0 < t < 1 along the path where t=0 is the start point and t=1 is the end point of the loops.
                    `phi`: the azimuthal angle of the loops' normal vector (default: 0.0°)
                    `theta`: the elevation angle of the loops' normal vector (default: 22.5°)
                    `r`: the radius of the loops (default: such that elevation at the top of the loops is 45°)
            '''
            if phi is None:
                phi = 0.0
            if theta is None:
                theta = self.theta
            if r is None:
                r = self.r
            # print('phi, theta, r =', phi, theta, r)
            # print(f't = {t}')
            # Rotation matrix around the azimuth
            rot_mat = np.array([
                [np.cos(phi), -np.sin(phi), 0.],
                [np.sin(phi), np.cos(phi), 0.],
                [0., 0., 1.]])
            # Normal vector of loop: points from center of loop to center of sphere
            n = rot_mat.dot(np.array([-np.cos(theta), 0., -np.sin(theta)]))
            # print(f'n = {n} {n.shape}')
            # Center of loop
            c = -n * np.sqrt(self.R**2 - r**2)
            # Second orthogonal vector in loop plane: points from loop center to the left (from pilot's POV).
            # We define this one first because it's easy.
            v = rot_mat.dot(np.array([0., 1., 0.]))
            # First orthogonal vector in loop plane: points from loop center down to the starting point of figure
            u = np.cross(n, v)

            # The resulting point
            # if isinstance(t, (list, np.ndarray)):
            p = np.array([c + r * (np.cos(k * t_i) * u + np.sin(k * t_i) * v) for t_i in t])
            # print(p.shape)
            # else:
            #     p = c + r * (np.cos(k * t) * u + np.sin(k * t) * v)

            # print(f'azimuth = {azimuth}')
            # print(f't = {t}')
            # print(f'c = {c}')
            # print(f'n = {n}')
            # print(f'u = {u}')
            # print(f'v = {v}')
            # print(f'c - p = {c - p}')
            # print()
            return p

        return func


def test():
    diags_on = True
    data = np.load(
        r'../2020-08-12 canandaigua [field markers]/001_20200812181538_fig0_inside_loop_out_figures_orig.npz'
        # r'../2020-08-12 canandaigua [field markers]/002_20200812183118_fig0_inside_loop_figures_orig.npz'
    )
    # fig_actuals = data['fig0'][:, 0, :]
    fig_actuals = data['fig0']
    print(f'shape of fig_actuals: {fig_actuals.shape}')

    trimmed_actuals = fig_actuals
    # trims entry/exit points.
    # trimmed_actuals = fig_actuals[27:493]
    # trimmed_actuals = fig_actuals[:170]

    fig0 = Figure.create(FigureTypes.INSIDE_LOOPS, actuals=trimmed_actuals, enable_diags=diags_on)
    # print()
    p_fig0 = fig0.fit()

    print(f'  args where t < 0: {fig0.diag.args_low}')
    print(f'  args where t > 1: {fig0.diag.args_high}')
    print(f'shape of initial u: {fig0.diag.u0.shape}')
    print(f'  shape of final u: {fig0.u.shape}')

    if diags_on:
        plt.show()

    assert fig0.R == common.DEFAULT_FLIGHT_RADIUS, 'Unexpected default radius R'

    print('=' * 160)
    t = np.linspace(0., 1., 8)
    test_pts = fig0.get_nom_point(*p_fig0, *t)
    norms = np.linalg.norm(test_pts, axis=1)
    print(norms)
    assert np.allclose(norms, fig0.R), "Unexpected distances of test points from sphere center"

    # phi = np.radians(90.0)
    # test_pts = [
    #     fig0.get_nom_point(0.0, phi),  # start
    #     fig0.get_nom_point(0.5 / 7.0, phi),  # first quarter-loop
    #     fig0.get_nom_point(1.0 / 7.0, phi),  # first half-loop
    #     fig0.get_nom_point(2.0 / 7.0, phi),  # first full loop
    #     fig0.get_nom_point(1.0, phi),  # end
    # ]
    # # print('test_pts:')
    # for test_pt in test_pts:
    #     # print(test_pt)
    #     assert np.linalg.norm(test_pt) == fig0.R, \
    #         "Unexpected distance of test point from sphere center"


if __name__ == '__main__':
    test()
