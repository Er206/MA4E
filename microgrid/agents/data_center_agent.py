import datetime
from microgrid.environments.data_center.data_center_env import DataCenterEnv
import sys
import os
import pulp
import pandas as pnd



class DataCenterAgent:
    def __init__(self, env: DataCenterEnv):
        self.env = env

    def take_decision(self, state):
        def data_center():
            L_IT = state['consumption_prevision']
            dt = self.env.delta_t / datetime.timedelta(hours=1)
            C1 = (self.env.COP_CS / self.env.EER) * (1 / (dt * (self.env.COP_HP - 1)))
            L_NF = (1 + 1 / (self.env.EER * dt)) * L_IT
            Lambda = state["manager_signal"]
            pwh = state["hotwater_price_prevision"]
            lp = pulp.LpProblem("DataCenter.lp", pulp.LpMinimize)
            lp.setSolver()
            alpha = [0 for i in range(48)]
            for t in range(48):
                # creation des variables
                ###########################################################
                var_name = "alpha" + str(t)
                alpha[t] = pulp.LpVariable(var_name, 0.0, 1.0)

                # creation des contraintes (a adapter)
                ###########################################################
                constraint_name = "Borne sup" + str(t)
                lp += alpha[t] <= (self.env.max_transfert) / (dt * self.env.COP_HP * C1 * L_IT[t]), constraint_name
                constraint_name = "positive" + str(t)
                lp += alpha[t] >= 0
                constraint_name = "inférieure à 1" + str(t)
                lp += alpha[t] <= 1

            # creation de la fonction objectif
            ###########################################################
            lp.setObjective(pulp.lpSum([Lambda[t] * (L_NF[t] + alpha[t] * C1 * L_IT[t]) * dt - (
                        self.env.COP_HP * dt * C1 * L_IT[t] * alpha[t] * pwh[t]) for t in range(48)]))
            return lp

        def run():
            I = data_center()
            I.solve()
            R = [1 for i in range(48)]
            for v in I.variables():
                print(v.name, "=", v.varValue)
                R[int(v.name[5:])] = v.varValue
            return R

        return run()

if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    data_center_config = {
        'scenario': 10,
    }
    env = DataCenterEnv(data_center_config, nb_pdt=N)
    agent = DataCenterAgent(env)
    cumulative_reward = 0
    now = datetime.datetime.now()
    state = env.reset(now, delta_t)
    for i in range(N*2):
        action = agent.take_decision(state)
        state, reward, done, info = env.step(action)
        cumulative_reward += reward
        if done:
            break
        print(f"action: {action}, reward: {reward}, cumulative reward: {cumulative_reward}")
        print("State: {}".format(state))