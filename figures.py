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
import enum
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt  # for debug diagnostics
import numpy as np
from scipy import optimize


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
    def __init__(self):
        self.args_low = None
        self.args_high = None
        self.errs_hist = None
        self.fit_params = None
        self.u0 = None
        self.trim_indexes = None


@enum.unique
class FigureTypes(enum.Enum):
    TAKEOFF = 1
    REVERSE_WINGOVER = 2
    INSIDE_LOOPS = 3
    INVERTED_FLIGHT = 4
    OUTSIDE_LOOPS = 5
    INSIDE_SQUARE_LOOPS = 6
    OUTSIDE_SQUARE_LOOPS = 7
    INSIDE_TRIANGULAR_LOOPS = 8
    HORIZONTAL_EIGHTS = 9
    HORIZONTAL_SQUARE_EIGHTS = 10
    VERTICAL_EIGHTS = 11
    HOURGLASS = 12
    OVERHEAD_EIGHTS = 13
    FOUR_LEAF_CLOVER = 14
    LANDING = 15


class Figure:
    '''Base class for all F2B figures.'''
    DEFAULT_FLYING_RADIUS = 21.0

    def __init__(self, R=None, actuals=None):
        '''Create a figure in a sphere of radius R associated with given actual points.'''
        super().__init__()
        if R is None:
            self.R = Figure.DEFAULT_FLYING_RADIUS
        else:
            self.R = R
        self._actuals = actuals
        if self._actuals is not None:
            act_centroid = self._actuals.mean(axis=0)
            # print(f'\n\nact_centroid = {act_centroid}\n\n')
            # Azimuth of the centroid of the actual points (rough estimate)
            self.phi0 = np.arctan2(act_centroid[1], act_centroid[0])
        else:
            self.phi0 = 0.
        self.diag = FigureDiagnostics()
        # Suggested count of nominal points. Override in subclasses as needed. This is primarily used for drawing.
        self.num_nominal_pts = 100
        # Initial t-parameterization
        self._init_path()
        # Reference to each subclassed Figure's parametric function
        self.paramfunc = self._parmfn()

    @staticmethod
    def create(which_figure, R=None, actuals=None):
        '''Factory method for creating a given Figure on sphere of radius R
        with specified actual path points.'''
        mapper = {
            FigureTypes.INSIDE_LOOPS: InsideLoops,
        }
        fig_type_constructor = mapper.get(which_figure, None)
        if fig_type_constructor:
            return fig_type_constructor(R=R, actuals=actuals)
        else:
            raise NotImplementedError(
                f'*** T O D O ***: Figure type {which_figure} is not implemented.')

    def fit(self, plot=False):
        '''Perform best fit of actual points against the nominals of the figure.'''

        errs = defaultdict(list)
        fit_idx = 0

        def get_ext_residuals(p, points, u):
            '''Objective function for least-squares fit'''
            # print('in get_ext_residuals')
            nonlocal errs
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
            # eps = sum(dists)
            eps = dists
            # print(f'eps = {eps}, p = {p}')
            errs[fit_idx].append(dists)
            return eps

        def get_int_residuals(u, points, p):
            '''Objective function for optimizing the internal t values along the path.'''
            # print('in get_int_residuals')
            nonlocal errs
            nonlocal fit_idx
            p_u = self.paramfunc(*p, *u)
            diff = points - p_u
            dists = np.linalg.norm(diff, axis=1)
            eps = dists
            errs[fit_idx].append(dists)
            return eps

        fit_params = {}
        fit_errs = {}

        c = itertools.cycle('!@#@#@#@#@#@#@#@#@#@#@#@#')
        extrinsics = []
        for fit_idx in range(3):
            print(next(c) * 80)
            if fit_idx % 2 == 0:
                print(f'Starting fit #{fit_idx}: external figure parameters...')
                if fit_idx == 0:
                    x0 = self.p0
                    arg2 = self.u
                else:
                    x0 = fit_params[fit_idx - 2]
                    arg2 = fit_params[fit_idx - 1]
                fit_params[fit_idx], fit_errs[fit_idx] = optimize.leastsq(
                    get_ext_residuals, x0, args=(self._actuals, arg2))
                print(f'Fit #{fit_idx} done, results = {fit_params[fit_idx], fit_errs[fit_idx]}\n')
                extrinsics.append(fit_params[fit_idx])
            else:
                if fit_idx == 1:
                    x0 = self.u
                else:
                    x0 = fit_params[fit_idx - 2]
                arg2 = fit_params[fit_idx - 1]
                print(f'Starting fit #{fit_idx}: internal t parameters...')
                fit_params[fit_idx], fit_errs[fit_idx] = optimize.leastsq(
                    get_int_residuals, x0, args=(self._actuals, arg2))
                print(f'Fit #{fit_idx} done, results:')
                # {fit_params[fit_idx], fit_errs[fit_idx]}\n')
                self.diag.args_low = list(np.argwhere(fit_params[fit_idx] < 0.0).flat)
                self.diag.args_high = list(np.argwhere(fit_params[fit_idx] > 1.0).flat)
                print(f'args where t < 0: {self.diag.args_low}')
                print(f'args where t > 1: {self.diag.args_high}')

        # Histories of the fitting journey (for diagnostics only). We save one per each get_*_residuals() call.
        self.diag.errs_hist = errs  # history of residual sets

        if plot:
            for f_idx, f_errs in errs.items():
                n_errs = len(f_errs)
                print(f'f_idx={f_idx}, n_errs={n_errs}')
                if n_errs == 0:
                    continue
                plt.figure(f_idx)
                for i, err in enumerate(f_errs):
                    if 0 < i < n_errs:
                        plt.plot(err, color='tab:gray', marker=None, linestyle='--')
                plt.plot(f_errs[0], 'r.-')
                plt.plot(f_errs[-1], 'b.-')
                plt.title(f'Errors during fit #{f_idx} ({n_errs} iterations)')

            plt.figure(f_idx + 1)
            plt.plot(self.u, 'r.-')
            plt.plot(fit_params[1], 'b.-')
            plt.plot((0, len(self.u)), (0., 0.), 'r--')
            plt.plot((0, len(self.u)), (1., 1.), 'r--')
            plt.title('Initial and final $u_i$')

            plt.figure(f_idx + 2)
            extrinsics = np.asarray(extrinsics)
            plt.plot(np.diff(extrinsics, axis=0), '.-')
            plt.title('Trends: extrinsic parameters at each iteration')

            plt.show()

        self.diag.fit_params = fit_params
        self.diag.u0 = self.u.copy()
        candidates = fit_params[1]
        self.diag.trim_indexes = np.logical_and(candidates >= 0.0, candidates <= 1.0)
        self.u = candidates[self.diag.trim_indexes]
        return fit_params[fit_idx], fit_errs[fit_idx]

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
    '''Represents three consecutive inside loops per F2B Rule 4.2.15.5 and Diagram 4.J.3 in the Annex.'''

    def __init__(self, R=None, actuals=None):
        super().__init__(R=R, actuals=actuals)
        # Nominal point every 5 degrees of the loop
        self.num_nominal_pts = 252
        # Initial guesses for optimization params
        self.p0 = [self.phi0, self.theta, self.r]
        # print(f'self.p0 = {self.p0}')
        # self.p0 = [0., 0., 1.] # rather extreme case, just to test sensitivity/convergence

    def fit(self, plot=False):
        '''Fit actual flight path to this Figure's nominal path.'''

        print('Initial parameters:')
        print(f'  phi   = {self.p0[0]} ({np.degrees(self.p0[0]):.3f}°)')
        print(f'  theta = {self.p0[1]} ({np.degrees(self.p0[1]):.3f}°)')
        print(f'  r     = {self.p0[2]}')

        p_fit, fit_err = super().fit(plot=plot)

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
            c = -n * self.R * np.cos(0.5 * alpha)
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
    plot_me = 1
    data = np.load(
        r'../2020-08-12 canandaigua [field markers]/002_20200812183118_fig0_inside_loop_figures__orig.npz')
    fig_actuals = data['fig0'][:, 0, :]
    # print(fig_actuals.shape)

    trimmed_actuals = fig_actuals
    # trims entry/exit points. TODO: do this programmatically, maybe RANSAC is a good algo for this?
    # trimmed_actuals = fig_actuals[27:493]

    fig0 = Figure.create(FigureTypes.INSIDE_LOOPS, actuals=trimmed_actuals)
    # print()
    p_fig0 = fig0.fit(plot=plot_me)

    print(f'  args where t < 0: {fig0.diag.args_low}')
    print(f'  args where t > 1: {fig0.diag.args_high}')
    print(f'shape of initial u: {fig0.diag.u0.shape}')
    print(f'  shape of final u: {fig0.u.shape}')

    if plot_me:
        plt.show()

    assert fig0.R == Figure.DEFAULT_FLYING_RADIUS, 'Unexpected default radius R'

    print('=' * 400)
    t = np.linspace(0., 1., 8)
    test_pts = fig0.get_nom_point(None, None, None, *t)
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
