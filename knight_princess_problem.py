import numpy as np
import random
from copy import deepcopy


class KnightPrincess:
    def __init__(self, states=(8, 8), actions=4):
        self.epsilon = 1.
        self.learning_rate = 0.5
        self.discount_rate = 0.9
        self.curr_state = [0, 0]

        if type(states) == tuple:
            self.q_table = np.zeros(list(states + (actions,)))
        else:
            self.q_table = np.zeros([states, actions])

        self.action_info = {0: (0, -1), 1: (-1, 0), 2: (0, 1), 3: (1, 0)}

    def run_episodes(self):
        print(self.q_table)
        for i in range(10000):
            self.episode()

        print(self.q_table)

    def reset_episode(self):
        self.curr_state = [0, 0]

    def episode_end(self):
        if self.curr_state in [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1],
                               [6, 2], [6, 3], [6, 4], [6, 6], [5, 6], [4, 6], [3, 6], [2, 6], [4, 4]]:
            return True
        return False

    def reward(self):
        if self.curr_state in [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1],
                               [6, 2], [6, 3], [6, 4], [6, 6], [5, 6], [4, 6], [3, 6], [2, 6]]:
            return -100
        elif self.curr_state == [4, 4]:
            print("happened")
            return 1000
        else:
            return -1

    def episode(self):
        no_steps = 0
        while not self.episode_end():
            action = self.select_action()
            x, y = self.action_info[action]

            old_state = deepcopy(self.curr_state)
            self.curr_state = [self.curr_state[0] + x, self.curr_state[1] + y]

            no_steps += 1

            if no_steps == 200:
                self.q_table[old_state[0], old_state[1], action] += (self.learning_rate * (-1000
                            + (self.discount_rate * np.amax(self.q_table[self.curr_state[0], self.curr_state[1]]))
                            - self.q_table[old_state[0], old_state[1], action]))
                break

            self.q_table[old_state[0], old_state[1], action] += (self.learning_rate * (self.reward()
                            + (self.discount_rate * np.amax(self.q_table[self.curr_state[0], self.curr_state[1]]))
                            - self.q_table[old_state[0], old_state[1], action]))

        self.epsilon -= (0.0005 * self.epsilon)
        self.reset_episode()

    def test_episode(self):
        self.epsilon = 0.01
        score = 0
        while not self.episode_end():
            action = self.select_action()
            x, y = self.action_info[action]
            print(self.curr_state, x, y)
            self.curr_state = [self.curr_state[0] + x, self.curr_state[1] + y]
            score += self.reward()
        print(self.curr_state)
        print(score)

    def select_action(self):
        rand = random.random()

        if rand > self.epsilon:
            temp = self.q_table[self.curr_state[0], self.curr_state[1]].argsort()
            for i in range(len(temp)):
                action = temp[len(temp) - i - 1]
                x, y = self.action_info[action]
                if (self.curr_state[0] + x >= 0) and (self.curr_state[0] + x < np.shape(self.q_table)[0]) \
                        and (self.curr_state[1] + y >= 0) and (self.curr_state[1] + y < np.shape(self.q_table)[1]):
                    break

        else:
            action = random.randint(0, 3)
            x, y = self.action_info[action]

            while not( (self.curr_state[0] + x >= 0) and (self.curr_state[0] + x < np.shape(self.q_table)[0]) \
                        and (self.curr_state[1] + y >= 0) and (self.curr_state[1] + y < np.shape(self.q_table)[1])):
                action = random.randint(0, 3)
                x, y = self.action_info[action]

        return action


if __name__ == "__main__":
    kpProblem = KnightPrincess()
    kpProblem.run_episodes()

    kpProblem.test_episode()
