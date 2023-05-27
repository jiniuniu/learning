import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Simulator:
    # 洞的半径相较于墙的尺度的比例
    hole_ratio: float = 0.1
    # 小球数
    num_particles: int = 300
    # 空间尺度
    rscale: float = 8.0e6
    # 时间尺度
    tscale: float = 1e9
    # 30帧每秒
    frame_per_sec: float = 30

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
        # dt
        self.dt = 1 / self.frame_per_sec

    def step(self, dt):
        """Advance the simulation by dt seconds."""

        self.pos += self.vel * dt

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

            # self.vel[i] = vel_j
            # self.vel[j] = vel_i

        # 对于撞到边界，该方向的速度分量取反
        hit_left_wall = self.pos[:, 0] < self.r
        hit_right_wall = self.pos[:, 0] > 3 - self.r
        hit_bottom_wall = self.pos[:, 1] < self.r
        hit_top_wall = self.pos[:, 1] > 1 - self.r
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


if __name__ == "__main__":
    sim = Simulator()
    sim.init_environment()
    n = sim.num_particles
    hole_r = sim.hole_ratio
    dt = sim.dt

    # Figure 相关的参数
    DPI = 100
    width, height = 1000, 600
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.subplots_adjust(left=0.1, right=0.97)
    sim_ax = fig.add_subplot(211, xlim=[0, 3], ylim=[0, 1], autoscale_on=False)
    sim_ax.set_xticks([])
    sim_ax.set_yticks([])
    # Make the box walls a bit more substantial.
    for spine in sim_ax.spines.values():
        spine.set_linewidth(2)

    npart_ax = fig.add_subplot(212)
    npart_ax.set_xlabel("Time, $t\;/\mathrm{ns}$")
    npart_ax.set_ylabel("Number of particles")
    npart_ax.set_xlim(0, 100)
    npart_ax.set_ylim(0, n)
    npart_ax.axhline(n / 3, 0, 1, color="k", lw=1, ls="--")

    (particles,) = sim_ax.plot([], [], "o", color="k")
    sim_ax.vlines(1, 0, 0.5 - hole_r, lw=2, color="k")
    sim_ax.vlines(1, 0.5 + hole_r, 1, lw=2, color="k")
    sim_ax.vlines(2, 0, 0.5 - hole_r, lw=2, color="k")
    sim_ax.vlines(2, 0.5 + hole_r, 1, lw=2, color="k")
    sim_ax.axvspan(0, 1, 0.0, 1, facecolor="tab:blue", alpha=0.3)
    sim_ax.axvspan(1, 2, 0.0, 1, facecolor="tab:red", alpha=0.3)
    sim_ax.axvspan(2, 3, 0.0, 1, facecolor="tab:green", alpha=0.3)

    blue_label_pos = 0.25, 1.05
    blue_label = sim_ax.text(*blue_label_pos, "Blue: {:d}".format(n), ha="center")
    red_label_pos = 1.25, 1.05
    red_label = sim_ax.text(*red_label_pos, "Red: 0", ha="center")
    green_label_pos = 2.25, 1.05
    green_label = sim_ax.text(*green_label_pos, "Green: 0", ha="center")

    (green_line,) = npart_ax.plot(
        [0], [0], c="g", label="number of particles in green box"
    )
    (red_line,) = npart_ax.plot([0], [0], c="r", label="number of particles in red box")
    (blue_line,) = npart_ax.plot(
        [0], [n], c="b", label="number of particles in blue box"
    )

    t, blues, reds, greens = [], [], [], []

    def animate(i):
        global sim
        sim.step(dt)

        particles.set_data(sim.pos[:, 0], sim.pos[:, 1])
        particles.set_markersize(2)

        t.append(i * dt)
        blue = sum(sim.pos[:, 0] < 1)
        red = sum(sim.pos[:, 0] < 2) - blue
        green = n - blue - red
        blues.append(blue)
        reds.append(red)
        greens.append(green)

        blue_label.set_text("Blue: {:d}".format(blue))
        red_label.set_text("Red: {:d}".format(red))
        green_label.set_text("Green: {:d}".format(green))

        green_line.set_data(t, greens)
        red_line.set_data(t, reds)
        blue_line.set_data(t, blues)

        return particles, blue_label, red_label, green_label

    nframes = 3000
    anim = FuncAnimation(fig, animate, frames=nframes, interval=10, repeat=False)
    # anim.save("diffusion_point_mass_approximation.mp4")
    plt.show()
