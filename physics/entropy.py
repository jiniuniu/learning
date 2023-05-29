import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List


class Simulator:
    # 洞的半径相较于墙的尺度的比例
    hole_ratio: float = 0.1
    # 小球数
    num_particles: int = 1000
    # 空间尺度
    rscale: float = 8.0e6
    # 时间尺度
    tscale: float = 1e9
    # 30帧每秒
    frame_per_sec: float = 30

    # grid 的尺度，~2 表示一条边用两个grid来表示
    grids: List[int] = [2, 12, 20]

    def init_environment(self):
        # 初始化，位置在空间均匀，速度在方向均匀分布
        self.pos = np.random.random((self.num_particles, 2)) * (1, 1)
        # 平均速度 ~150 m.s-1.
        sbar = 150 * self.rscale / self.tscale
        theta = np.random.random(self.num_particles) * 2 * np.pi
        s0 = sbar * np.random.random(self.num_particles)
        self.vel = (s0 * np.array((np.cos(theta), np.sin(theta)))).T
        # van der Waals radius of Ar, ~0.2 nm.
        self.r = 20e-10 * self.rscale
        # ~每次系统evolve的时间步长
        self.dt = 1 / self.frame_per_sec

    def step(self):
        """Advance the simulation by dt seconds."""

        self.pos += self.vel * self.dt

        # dist[i][j] 表示小球 i 和 j 的距离，找到碰撞的小球
        dist = squareform(pdist(self.pos))
        collision_i, collision_j = np.where(dist < 2 * self.r)
        k = collision_i < collision_j
        collision_i, collision_j = collision_i[k], collision_j[k]

        # 刚体碰撞速度公式，考虑弹性碰撞以及相同质量做简化
        # https://github.com/phenomLi/Blog/issues/35
        for i, j in zip(collision_i, collision_j):
            pos_i, vel_i = self.pos[i], self.vel[i]
            pos_j, vel_j = self.pos[j], self.vel[j]
            r_ij, v_ij = pos_i - pos_j, vel_i - vel_j

            r_dot_r = r_ij @ r_ij
            v_dot_r = v_ij @ r_ij
            Jn = -v_dot_r * r_ij / r_dot_r
            self.vel[i] += Jn
            self.vel[j] -= Jn

        # 对于撞到边界，该方向的速度分量取反
        hit_left_wall = self.pos[:, 0] < self.r
        hit_right_wall = self.pos[:, 0] > 2 - self.r
        hit_bottom_wall = self.pos[:, 1] < self.r
        hit_top_wall = self.pos[:, 1] > 1 - self.r
        self.pos[hit_left_wall, 0] = self.r
        self.pos[hit_right_wall, 0] = 2 - self.r
        self.pos[hit_bottom_wall, 1] = self.r
        self.pos[hit_top_wall, 1] = 1 - self.r
        self.vel[hit_left_wall | hit_right_wall, 0] *= -1
        self.vel[hit_bottom_wall | hit_top_wall, 1] *= -1

        hit_middle_wall = (
            (self.pos[:, 1] < (0.5 - self.hole_ratio))
            | (self.pos[:, 1] > (0.5 + self.hole_ratio))
        ) & (
            ((self.pos[:, 0] > (1 - self.r)) & (self.pos[:, 0] < (1 + self.r)))
            | ((self.pos[:, 0] > (2 - self.r)) & (self.pos[:, 0] < (2 + self.r)))
        )
        self.vel[hit_middle_wall, 0] *= -1

    def get_entropy(self, grid: int):
        """
        "Classical dynamical coarse-grained entropy and comparison with
        the quantum version." Physical Review E 102.3 (2020): 032106.
        """
        pis = np.zeros(grid**2 * 2)
        for bix in range(grid * 2):
            for biy in range(grid):
                ci = bix * grid + biy
                x1 = bix / grid
                x2 = (bix + 1) / grid
                y1 = biy / grid
                y2 = (biy + 1) / grid
                nwherex = np.where(
                    np.logical_and(self.pos[:, 0] > x1, self.pos[:, 0] < x2)
                )[0]
                nwherey = np.where(
                    np.logical_and(self.pos[:, 1] > y1, self.pos[:, 1] < y2)
                )[0]
                nbi = len(np.intersect1d(nwherex, nwherey))
                pis[ci] = nbi / self.num_particles

        pis_ = pis[pis > 0]
        return -np.sum(pis_ * np.log(pis_))


if __name__ == "__main__":
    sim = Simulator()
    sim.init_environment()
    n = sim.num_particles
    hole_r = sim.hole_ratio
    dt = sim.dt

    # Figure 相关的参数
    DPI = 100
    width, height = 600, 600
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.97)
    sim_ax = fig.add_subplot(211, xlim=[0, 2], ylim=[0, 1], autoscale_on=False)
    sim_ax.set_xticks([])
    sim_ax.set_yticks([])
    # Make the box walls a bit more substantial.
    for spine in sim_ax.spines.values():
        spine.set_linewidth(2)

    entropy_ax = fig.add_subplot(212)
    entropy_ax.set_xlabel("Time, $t\;/\mathrm{ns}$")
    entropy_ax.set_ylabel("entropy")
    # 热平衡均匀分布下，熵最大。
    s_max = np.log(sim.grids[0] ** 2 * 2)
    s_min = np.log(sim.grids[0] ** 2)
    entropy_ax.set_xlim(0, 100)
    entropy_ax.set_ylim(-0.1, (s_max - s_min) * 1.2)
    entropy_ax.axhline(s_max - s_min, 0, 1, color="k", lw=1, ls="--")

    (particles,) = sim_ax.plot([], [], "o", color="k")
    sim_ax.vlines(1, 0, 0.5 - hole_r, lw=2, color="k")
    sim_ax.vlines(1, 0.5 + hole_r, 1, lw=2, color="k")
    sim_ax.axvspan(0, 1, 0.0, 1, facecolor="tab:blue", alpha=0.3)
    sim_ax.axvspan(1, 2, 0.0, 1, facecolor="tab:red", alpha=0.3)

    (red_line,) = entropy_ax.plot([0], [0], c="r", label="4 x 2 grids")
    (blue_line,) = entropy_ax.plot([0], [0], c="b", label="24 x 12 grids")
    (green_line,) = entropy_ax.plot([0], [0], c="g", label="40 x 20 grids")
    entropy_ax.legend()

    s0 = []
    for grid in sim.grids:
        s0.append(np.log(grid**2))

    t, blues, reds, greens = [], [], [], []

    def animate(i):
        global sim
        sim.step()

        particles.set_data(sim.pos[:, 0], sim.pos[:, 1])
        particles.set_markersize(2)

        t.append(i * dt)

        reds.append(sim.get_entropy(sim.grids[0]) - s0[0])
        blues.append(sim.get_entropy(sim.grids[1]) - s0[1])
        greens.append(sim.get_entropy(sim.grids[2]) - s0[2])

        red_line.set_data(t, reds)
        blue_line.set_data(t, blues)
        green_line.set_data(t, greens)

        return particles

    nframes = 3000
    anim = FuncAnimation(fig, animate, frames=nframes, interval=10, repeat=False)
    # anim.save("entropy_demo.mp4")
    plt.show()
