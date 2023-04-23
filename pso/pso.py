import numpy as np


class Particle:
    def __init__(self, initial_pos) -> None:
        self.pos = initial_pos
        self.fitness = np.inf
        self.p_best = [self.pos, np.inf]
        self.vel = np.zeros(2)

    def update_fitness(self, next_fitness):
        if next_fitness < self.p_best[1]:
            self.p_best = [self.pos, next_fitness]
        self.fitness = next_fitness

    def __iter__(self):
        yield self.pos
        yield self.vel
        yield self.p_best


class Swarm:
    def __init__(
        self,
        size,
        x_bounds,
        y_bounds,
        w,
        c1,
        c2,
        velocity_factor=0.1,
        random_state=None,
    ) -> None:
        self.rand = np.random.RandomState(random_state or 42)
        self.w, self.c1, self.c2 = w, c1, c2
        self.vel_factor = velocity_factor

        self.particles = []
        for _ in range(size):
            pos_x = self.rand.uniform(x_bounds[0], x_bounds[1])
            pos_y = self.rand.uniform(y_bounds[0], y_bounds[1])
            self.particles.append(Particle(np.array([pos_x, pos_y])))

        self.g_best = [self.particles[0].pos, self.particles[0].fitness]
        self._calc_mean_fitness()

    def _calc_mean_fitness(self):
        self.mean_fitness = np.mean([particle.fitness for particle in self.particles])

    def calc_fitness(self, fit_func):
        for particle in self.particles:
            d = fit_func(particle.pos)
            particle.update_fitness(d)
            if d < self.g_best[1]:
                self.g_best = [particle.pos, d]
        self._calc_mean_fitness()

    def step(self):
        for particle in self.particles:
            g_best = self.g_best[0]
            p_best = particle.p_best[0]

            v1 = (p_best - particle.pos) * self.c1 * self.rand.random()
            v2 = (g_best - particle.pos) * self.c2 * self.rand.random()

            particle.vel = self.vel_factor * (v1 + v2 + particle.vel * self.w)
            particle.pos = particle.pos + particle.vel

    def get_pos_values(self):
        xvals, yvals = [], []
        for particle in self.particles:
            xvals.append(particle.pos[0])
            yvals.append(particle.pos[1])
        return xvals, yvals
