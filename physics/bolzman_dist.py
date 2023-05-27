import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
from matplotlib.animation import FuncAnimation


class Simulator:
    # 小球数量
    num_particles: int = 1000
    # 空间尺度，box的边长即 1/rscale
    rscale: float = 5.0e6
    # 时间尺度
    tscale: float = 1e9
    # 30帧每秒
    frame_per_sec: float = 30
    # 质量 ~1.0
    m: float = 1.0

    def init_environment(self):
        # 初始化，位置在空间均匀，速度在方向均匀分布
        self.pos = np.random.random((self.num_particles, 2)) * (1, 1)
        # 平均速度 ~353 m.s-1. root-mean-square velocity of Ar at 300 K
        self.sbar = 353 * self.rscale / self.tscale
        theta = np.random.random(self.num_particles) * 2 * np.pi
        s0 = self.sbar * np.random.random(self.num_particles)
        self.vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
        # van der Waals radius of Ar, ~0.2 nm.
        self.r = 20e-10 * self.rscale
        # ~每次系统evolve的时间步长
        self.dt = 1 / self.frame_per_sec

    def step(self):
        self.pos += self.vel * self.dt

        # 找到碰撞的小球 (dist < 2 * self.r)
        # dist[i][j] 表示小球 i 和 j 的距离
        dist = squareform(pdist(self.pos))
        iarr, jarr = np.where(dist < 2 * self.r)
        k = iarr < jarr
        iarr, jarr = iarr[k], jarr[k]

        # 刚体碰撞速度公式，考虑弹性碰撞以及相同质量做简化
        # https://github.com/phenomLi/Blog/issues/35
        for i, j in zip(iarr, jarr):
            pos_i, vel_i = self.pos[i], self.vel[i]
            pos_j, vel_j = self.pos[j], self.vel[j]
            r_ij, v_ij = pos_i - pos_j, vel_i - vel_j
            r_dot_r = r_ij @ r_ij
            v_dot_r = v_ij @ r_ij
            Jn = -v_dot_r * r_ij / r_dot_r
            self.vel[i] += Jn
            self.vel[j] -= Jn

        # 考虑撞墙的情况，延墙面的速度取反
        hit_left_wall = self.pos[:, 0] < self.r
        hit_right_wall = self.pos[:, 0] > 1 - self.r
        hit_bottom_wall = self.pos[:, 1] < self.r
        hit_top_wall = self.pos[:, 1] > 1 - self.r
        self.vel[hit_left_wall | hit_right_wall, 0] *= -1
        self.vel[hit_bottom_wall | hit_top_wall, 1] *= -1

    def get_speeds(self):
        return np.hypot(self.vel[:, 0], self.vel[:, 1])

    def get_kinetic_energy(self):
        return 0.5 * self.m * sum(self.get_speeds() ** 2)


class Histogram:
    """A class to draw a Matplotlib histogram as a collection of Patches."""

    def __init__(self, data, xmax, nbars, density=False):
        """Initialize the histogram from the data and requested bins."""
        self.nbars = nbars
        self.density = density
        self.bins = np.linspace(0, xmax, nbars)
        self.hist, bins = np.histogram(data, self.bins, density=density)

        # Drawing the histogram with Matplotlib patches owes a lot to
        # https://matplotlib.org/3.1.1/gallery/animation/animated_histogram.html
        # Get the corners of the rectangles for the histogram.
        self.left = np.array(bins[:-1])
        self.right = np.array(bins[1:])
        self.bottom = np.zeros(len(self.left))
        self.top = self.bottom + self.hist
        nrects = len(self.left)
        self.nverts = nrects * 5
        self.verts = np.zeros((self.nverts, 2))
        self.verts[0::5, 0] = self.left
        self.verts[0::5, 1] = self.bottom
        self.verts[1::5, 0] = self.left
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 0] = self.right
        self.verts[2::5, 1] = self.top
        self.verts[3::5, 0] = self.right
        self.verts[3::5, 1] = self.bottom

    def draw(self, ax):
        """Draw the histogram by adding appropriate patches to Axes ax."""
        codes = np.ones(self.nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        barpath = path.Path(self.verts, codes)
        self.patch = patches.PathPatch(
            barpath, fc="tab:green", ec="k", lw=0.5, alpha=0.5
        )
        ax.add_patch(self.patch)

    def update(self, data):
        """Update the rectangle vertices using a new histogram from data."""
        self.hist, bins = np.histogram(data, self.bins, density=self.density)
        self.top = self.bottom + self.hist
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 1] = self.top


if __name__ == "__main__":
    sim = Simulator()
    sim.init_environment()

    # 画图的参数
    DPI = 100
    width, height = 1000, 500
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.subplots_adjust(left=0, right=0.97)
    sim_ax = fig.add_subplot(121, aspect="equal", autoscale_on=False)
    sim_ax.set_xticks([])
    sim_ax.set_yticks([])
    # Make the box walls a bit more substantial.
    for spine in sim_ax.spines.values():
        spine.set_linewidth(2)

    speed_ax = fig.add_subplot(122)
    speed_ax.set_xlabel("Speed $v\,/m\,s^{-1}$")
    speed_ax.set_ylabel("$f(v)$")

    (particles,) = sim_ax.plot([], [], "ko")

    speeds = sim.get_speeds()
    speed_hist = Histogram(speeds, 2 * sim.sbar, 50, density=True)
    speed_hist.draw(speed_ax)
    speed_ax.set_xlim(speed_hist.left[0], speed_hist.right[-1])

    ticks = np.linspace(0, 600, 7, dtype=int)
    speed_ax.set_xticks(ticks * sim.rscale / sim.tscale)
    speed_ax.set_xticklabels([str(tick) for tick in ticks])
    speed_ax.set_yticks([])

    fig.tight_layout()

    # The 2D Maxwell-Boltzmann equilibrium distribution of speeds.
    mean_kinetic_engergy = sim.get_kinetic_energy() / sim.num_particles
    a = sim.m / 2 / mean_kinetic_engergy
    # Use a high-resolution grid of speed points so that the exact distribution
    # looks smooth.
    sgrid_hi = np.linspace(0, speed_hist.bins[-1], 200)
    f = 2 * a * sgrid_hi * np.exp(-a * sgrid_hi**2)
    (mb_line,) = speed_ax.plot(sgrid_hi, f, c="0.7")
    # Maximum value of the 2D Maxwell-Boltzmann speed distribution.
    fmax = np.sqrt(sim.m / mean_kinetic_engergy / np.e)
    speed_ax.set_ylim(0, fmax)

    # For the distribution derived by averaging, take the abcissa speed points from
    # the centre of the histogram bars.
    sgrid = (speed_hist.bins[1:] + speed_hist.bins[:-1]) / 2
    (mb_est_line,) = speed_ax.plot([], [], c="r")
    mb_est = np.zeros(len(sgrid))

    # A text label indicating the time and step number for each animation frame.
    xlabel, ylabel = sgrid[-1] / 2, 0.8 * fmax
    label = speed_ax.text(
        xlabel, ylabel, "$t$ = {:.1f}s, step = {:d}".format(0, 0), backgroundcolor="w"
    )

    # Only start averaging the speed distribution after frame number IAV_ST.
    IAV_START = 200
    # Number of frames; set to None to run until explicitly quit.
    frames = 1000

    def init_anim():
        """Initialize the animation"""
        particles.set_data([], [])

        return particles, speed_hist.patch, mb_est_line, label

    def animate(i):
        """Advance the animation by one step and update the frame."""
        global sim, mb_est_line, mb_est
        sim.step()

        particles.set_data(sim.pos[:, 0], sim.pos[:, 1])
        particles.set_markersize(0.5)

        speeds = sim.get_speeds()
        speed_hist.update(speeds)

        # Once the simulation has approached equilibrium a bit, start averaging
        # the speed distribution to indicate the approximation to the Maxwell-
        # Boltzmann distribution.
        if i >= IAV_START:
            mb_est += (speed_hist.hist - mb_est) / (i - IAV_START + 1)
            mb_est_line.set_data(sgrid, mb_est)

        label.set_text("$t$ = {:.1f} ns, step = {:d}".format(i * sim.dt, i))

        return particles, speed_hist.patch, mb_est_line, label

    anim = FuncAnimation(
        fig, animate, frames=frames, interval=10, blit=False, init_func=init_anim
    )
    plt.show()
