# MODİFİED FOR DRL HV1
# IN ORDER TO COMPLATE ALL TASKS ABOUT HV1 SOME FEATURES ADDED LİST BELOW
# MAP SİZE İNCREASED OLD =>5*5 NEW=>6*6
# STATİON SİZE İNCREASED OLS=>4 NEW=>5
# ADDED NE OBSTACLE NAMED X FOR RESTRİCT TAXİ MOVEMENT

# For the complate all off these changes I havo to change some part of taxi.py file
# For detailed changes make search in file as a "CHANGED"

from contextlib import closing
from io import StringIO
from os import path
from typing import Optional

import numpy as np

from gym import Env, logger, spaces, utils
from gym.envs.toy_text.utils import categorical_sample
from gym.error import DependencyNotInstalled


# CHANGED
# Whole render-logic system based on this value so I moddeled our new env(6*6 map, 5 station, obstacles etc.)
MAP = [
    "+-----------+",
    "|R: | : :G: |",
    "| : |X| : : |",
    "| : |X| : : |",
    "| : : : : : |",
    "| | : |P: : |",
    "|Y| : |B:X: |",
    "+-----------+",
]

WINDOW_SIZE = (750, 750)


class TaxiEnv(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.desc = np.asarray(MAP, dtype="c")
        # CHANGED
        # new station pos addedand color added
        self.locs = locs = [(0, 0), (0, 4), (4, 0), (4, 3), (5, 5)]
        self.locs_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (255, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (255, 255, 255),
        ]
        # CHANGED
        # based on map size and station size also effected total size of states (6*6*6*5)
        # base row and column values added
        num_states = 1080
        num_rows = 6
        num_columns = 6
        max_row = num_rows - 1
        max_col = num_columns - 1

        self.initial_state_distrib = np.zeros(num_states)
        num_actions = 6
        self.P = {
            state: {action: [] for action in range(num_actions)}
            for state in range(num_states)
        }

        # CHANGED-ADDED
        # NEW OBSTACLE(X) CHECK ADDED

        for row in range(num_rows):
            for col in range(num_columns):
                for pass_idx in range(len(locs) + 1):
                    for dest_idx in range(len(locs)):
                        state = self.encode(row, col, pass_idx, dest_idx)

                        is_obstacle = self.desc[1 + row, 2 * col + 1] == b"X"
                        # CHANGED
                        # because of new station
                        if pass_idx < 5 and pass_idx != dest_idx and not is_obstacle:
                            self.initial_state_distrib[state] += 1

                        for action in range(num_actions):
                            new_row, new_col, new_pass_idx = row, col, pass_idx
                            reward = -1
                            terminated = False
                            taxi_loc = (row, col)

                            if action == 0:
                                new_row = min(row + 1, max_row)
                            elif action == 1:
                                new_row = max(row - 1, 0)
                            if action == 2 and self.desc[1 + row, 2 * col + 2] == b":":
                                new_col = min(col + 1, max_col)
                            elif action == 3 and self.desc[1 + row, 2 * col] == b":":
                                new_col = max(col - 1, 0)

                            if self.desc[1 + new_row, 2 * new_col + 1] == b"X":
                                new_row, new_col = row, col

                            elif action == 4:
                                # CHANGED
                                # because of new station
                                if pass_idx < 5 and taxi_loc == locs[pass_idx]:
                                    new_pass_idx = 5
                                else:
                                    reward = -10
                            elif action == 5:
                                if (taxi_loc == locs[dest_idx]) and pass_idx == 5:
                                    new_pass_idx = dest_idx
                                    terminated = True
                                    reward = 20
                                elif (taxi_loc in locs) and pass_idx == 5:
                                    new_pass_idx = locs.index(taxi_loc)
                                else:
                                    reward = -10

                            new_state = self.encode(
                                new_row, new_col, new_pass_idx, dest_idx
                            )
                            self.P[state][action].append(
                                (1.0, new_state, reward, terminated)
                            )

        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(num_states)
        self.render_mode = render_mode

        self.window = None
        self.clock = None
        self.cell_size = (
            WINDOW_SIZE[0] / self.desc.shape[1],
            WINDOW_SIZE[1] / self.desc.shape[0],
        )
        self.taxi_imgs = None
        self.taxi_orientation = 0
        self.passenger_img = None
        self.destination_img = None
        self.median_horiz = None
        self.median_vert = None
        self.background_img = None

    # CHANGED
    # decode and encode func changed because of changed base values such as row column and station

    def encode(self, taxi_row, taxi_col, pass_loc, dest_idx):
        i = taxi_row
        i *= 6
        i += taxi_col
        i *= 6
        i += pass_loc
        i *= 5
        i += dest_idx
        return i

    def decode(self, i):
        out = []
        out.append(i % 5)
        i = i // 5
        out.append(i % 6)
        i = i // 6
        out.append(i % 6)
        i = i // 6
        out.append(i)
        assert 0 <= i < 6
        return reversed(out)

    def action_mask(self, state: int):
        mask = np.zeros(6, dtype=np.int8)
        taxi_row, taxi_col, pass_loc, dest_idx = self.decode(state)
        # CHANGED
        # because of new map size changes
        # also obstacle check added
        if taxi_row < 5 and self.desc[1 + taxi_row + 1, 2 * taxi_col + 1] != b"X":
            mask[0] = 1
        if taxi_row > 0 and self.desc[1 + taxi_row - 1, 2 * taxi_col + 1] != b"X":
            mask[1] = 1
        if (
            taxi_col < 5
            and self.desc[1 + taxi_row, 2 * taxi_col + 2] == b":"
            and self.desc[1 + taxi_row, 2 * (taxi_col + 1) + 1] != b"X"
        ):
            mask[2] = 1
        if (
            taxi_col > 0
            and self.desc[1 + taxi_row, 2 * taxi_col] == b":"
            and self.desc[1 + taxi_row, 2 * (taxi_col - 1) + 1] != b"X"
        ):
            mask[3] = 1

        if pass_loc < 5 and (taxi_row, taxi_col) == self.locs[pass_loc]:
            mask[4] = 1
        if pass_loc == 5 and (
            (taxi_row, taxi_col) == self.locs[dest_idx]
            or (taxi_row, taxi_col) in self.locs
        ):
            mask[5] = 1
        return mask

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        if self.render_mode == "human":
            self.render()
        return (int(s), r, t, False, {"prob": p, "action_mask": self.action_mask(s)})

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.s = categorical_sample(self.initial_state_distrib, self.np_random)
        self.lastaction = None
        self.taxi_orientation = 0

        if self.render_mode == "human":
            self.render()
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    def render(self):
        if self.render_mode is None:
            return
        if self.render_mode == "ansi":
            return self._render_text()
        else:
            return self._render_gui(self.render_mode)

    # CHANGED
    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled("pygame not installed")

        col_width = 64
        row_height = 64
        expected_width = self.desc.shape[1] * col_width
        expected_height = self.desc.shape[0] * row_height
        self.cell_size = (col_width, row_height)

        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Taxi 6x6 (Engelli)")
            if mode == "human":
                self.window = pygame.display.set_mode((expected_width, expected_height))
            elif mode == "rgb_array":
                self.window = pygame.Surface((expected_width, expected_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.taxi_imgs is None:
            file_names = [
                path.join(path.dirname(__file__), "img/cab_front.png"),
                path.join(path.dirname(__file__), "img/cab_rear.png"),
                path.join(path.dirname(__file__), "img/cab_right.png"),
                path.join(path.dirname(__file__), "img/cab_left.png"),
            ]
            self.taxi_imgs = [
                pygame.transform.scale(pygame.image.load(fn), self.cell_size)
                for fn in file_names
            ]
        if self.passenger_img is None:
            self.passenger_img = pygame.transform.scale(
                pygame.image.load(
                    path.join(path.dirname(__file__), "img/passenger.png")
                ),
                self.cell_size,
            )
        if self.destination_img is None:
            self.destination_img = pygame.transform.scale(
                pygame.image.load(path.join(path.dirname(__file__), "img/hotel.png")),
                self.cell_size,
            )
            self.destination_img.set_alpha(170)

        if self.median_horiz is None:
            fns = [
                "img/gridworld_median_left.png",
                "img/gridworld_median_horiz.png",
                "img/gridworld_median_right.png",
            ]
            self.median_horiz = [
                pygame.transform.scale(
                    pygame.image.load(path.join(path.dirname(__file__), fn)),
                    self.cell_size,
                )
                for fn in fns
            ]
        if self.median_vert is None:
            fns = [
                "img/gridworld_median_top.png",
                "img/gridworld_median_vert.png",
                "img/gridworld_median_bottom.png",
            ]
            self.median_vert = [
                pygame.transform.scale(
                    pygame.image.load(path.join(path.dirname(__file__), fn)),
                    self.cell_size,
                )
                for fn in fns
            ]
        if self.background_img is None:
            self.background_img = pygame.transform.scale(
                pygame.image.load(
                    path.join(path.dirname(__file__), "img/taxi_background.png")
                ),
                self.cell_size,
            )

        desc = self.desc

        for y in range(0, desc.shape[0]):
            for x in range(0, desc.shape[1]):
                cell = (x * self.cell_size[0], y * self.cell_size[1])

                self.window.blit(self.background_img, cell)
                # CHANGED - ADDED
                # OBSTACLES SHOWNS AS A BLACK SQUARE
                if desc[y][x] == b"X":
                    black_square = pygame.Surface(self.cell_size)
                    black_square.fill((0, 0, 0))
                    self.window.blit(black_square, cell)

                if desc[y][x] == b"|" and (y == 0 or desc[y - 1][x] != b"|"):
                    self.window.blit(self.median_vert[0], cell)
                elif desc[y][x] == b"|" and (
                    y == desc.shape[0] - 1 or desc[y + 1][x] != b"|"
                ):
                    self.window.blit(self.median_vert[2], cell)
                elif desc[y][x] == b"|":
                    self.window.blit(self.median_vert[1], cell)
                elif desc[y][x] == b"-" and (x == 0 or desc[y][x - 1] != b"-"):
                    self.window.blit(self.median_horiz[0], cell)
                elif desc[y][x] == b"-" and (
                    x == desc.shape[1] - 1 or desc[y][x + 1] != b"-"
                ):
                    self.window.blit(self.median_horiz[2], cell)
                elif desc[y][x] == b"-":
                    self.window.blit(self.median_horiz[1], cell)

        for cell, color in zip(self.locs, self.locs_colors):
            color_cell = pygame.Surface(self.cell_size)
            color_cell.set_alpha(128)
            color_cell.fill(color)
            loc = self.get_surf_loc(cell)
            self.window.blit(color_cell, (loc[0], loc[1]))

        taxi_row, taxi_col, pass_idx, dest_idx = self.decode(self.s)
        # CHANGED
        # because of new station
        if pass_idx < 5:
            self.window.blit(self.passenger_img, self.get_surf_loc(self.locs[pass_idx]))

        if self.lastaction in [0, 1, 2, 3]:
            self.taxi_orientation = self.lastaction

        dest_loc = self.get_surf_loc(self.locs[dest_idx])
        taxi_location = self.get_surf_loc((taxi_row, taxi_col))

        if dest_loc[1] <= taxi_location[1]:
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
        else:
            self.window.blit(self.taxi_imgs[self.taxi_orientation], taxi_location)
            self.window.blit(
                self.destination_img,
                (dest_loc[0], dest_loc[1] - self.cell_size[1] // 2),
            )

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )

    def get_surf_loc(self, map_loc):
        return (map_loc[1] * 2 + 1) * self.cell_size[0], (
            map_loc[0] + 1
        ) * self.cell_size[1]

    def _render_text(self):
        return ""
