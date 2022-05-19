import datetime
from microgrid.environments.solar_farm.solar_farm_env import SolarFarmEnv
import numpy as np


class SolarFarmAgent:
    def __init__(self, env: SolarFarmEnv):
        self.env = env

    def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):
     pb = pulp.LpProblem("ferme_solaire", pulp.LpMinimize)
        
        # variables (qu'on cherche à optimiser)
        l = {} #conso totale
        a = {} #charge batterie
        lbat = {} #conso batterie
        lbatpos = {} #conso batterie partie positive (ça sert pour la valeur absolue)
        lbatneg = {} #conso batterie négative
        signelbat = {} #signe de la conso
        
        
        # constantes
        dt = datetime.timedelta(minutes=30)
        T = self.env.nb_pdt #on va le chercher dans l'environnement
        
        
        #on va chercher les differentes valeurs des variables dans l'environnement
        
        rhoc = self.env.battery.efficiency #efficacité charge 
        rhod = self.env.battery.efficiency #eff décharge
        rhomax = self.env.battery.capacity #charge max batterie
        pmax = self.env.battery.pmax #puissance max
        
        surface = self.env.pv.surface

        lPV = 0.001 * surface * state['pv_prevision'] #prod parc pv
        a[0] = state['soc'] #charge batterie au début
        prix = state['manager_signal'] #lambda du tp

        #on énonce les variables

        for t in range(T):
            l[t] = pulp.LpVariable("l_" + str(t), None, None)
            lbat[t] = pulp.LpVariable("lbat_" + str(t), -pmax, pmax)
            lbatpos[t] = pulp.LpVariable("lbatpos_" + str(t), 0, pmax)
            lbatneg[t] = pulp.LpVariable("lbatneg_" + str(t), 0, pmax)
            signelbat[t] = pulp.LpVariable("signelbat_" + str(t), cat=pulp.LpBinary)
            a[t + 1] = pulp.LpVariable("a_" + str(t), 0, rhomax)

            #les contraintes
            pb += lbat[t] == lbatpos[t] - lbatneg[t], 
            pb += lbatpos[t] <= signelbat[t] * pmax, 
            pb += lbatneg[t] <= (1 - signelbat[t]) * pmax, 
            pb += l[t] == lbat[t] - lPV[t], 
            pb += a[t + 1] == a[t] + (self.env.battery.efficiency * lbatpos[t] - (1/self.env.battery.efficiency)*lbatneg[t])*dt, '''"charge_batterie_" + str(t)'''

        

        pb.setObjective(pulp.lpSum([conso_totale[t] * prix[t] * dt for t in range(T)])) #la fonction objectif pour minimiser le cout

        pb.solve()
        a = self.env.action_space.sample()
        for t in range(T):
            a[t]=l[t].varValue

        return a

if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    solar_farm_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'pv': {
            'surface': 100,
            'location': "enpc",  # or (lat, long) in float
            'tilt': 30,  # in degree
            'azimuth': 180,  # in degree from North
            'tracking': None,  # None, 'horizontal', 'dual'
        }
    }
    env = SolarFarmEnv(solar_farm_config=solar_farm_config, nb_pdt=N)
    agent = SolarFarmAgent(env)
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
        print("Info: {}".format(action.sum(axis=0)))
