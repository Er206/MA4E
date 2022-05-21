import datetime
from microgrid.environments.industrial.industrial_env import IndustrialEnv
import pulp

class IndustrialAgent:
    def __init__(self, env: IndustrialEnv):
        self.env = env

        def take_decision(self,
                      state,
                      previous_state=None,
                      previous_action=None,
                      previous_reward=None):

        #Données utiles:
        
        delta_t = datetime.timedelta(minutes=30)  #pas de temps
        H = datetime.timedelta(hours=1)
        T=self.env.nb_pdt #nombre de periodes temporelles
        capacity= self.env.battery.capacity  #capacité
        pmax=self.env.battery.pmax #puisance maximale
        efficiency=self.env.battery.efficiency #rendement

        manager_signal= state.get("manager_signal") #les prix  cf manager.py 
        consumption_prevision = state.get("consumption_prevision")  # la demande de consommation

        soc = state.get("soc")
        a_td = self.env.battery.soc

        #Problème:
        
        pb=pulp.LpProblem("Site industriel", pulp.LpMinimize)

        #Variables:
        
        l_bat_pos = pulp.LpVariable.dicts("l_bat_pos", [t for t in range(T)], pmax)  #l_bat+ <= pmax
        l_bat_neg = pulp.LpVariable.dicts("l_bat_neg", [t for t in range(T)], pmax)  #l_bat- <= pmax
        l_bat = pulp.LpVariable.dicts("l_bat", [t for t in range(T)])
        li = pulp.LpVariable.dicts("li", [t for t in range(T)])  #demande totale du site industriel (ie en incluant la présence de la batterie)
        a = pulp.LpVariable.dicts("stock_batterie", [t for t in range(T)], 0, capacity) #cf formulation mathématique  0 <= a <= C

        #fonction objectifll

        pb += pulp.lpSum([li[t] * manager_signal[t] * delta_t/H  for t in range(T)])

        #Contraintes

        pb += a[0]== a_td

        for t in range(T):
            pb += li[t] - consumption_prevision[t] - l_bat[t] == 0  #li(t)=ldem(t)+lbat(t)
            pb += l_bat[t] - (l_bat_pos[t] - l_bat_neg[t]) == 0

        for t in range(1,T):
            pb += a[t]-a[t-1]- (efficiency*l_bat_pos[t] - 1/efficiency*l_bat_neg[t])*delta_t/H ==0  #formule de recurrence a(t)


        #Résolution
        pb.solve()

        resultat = self.env.action_space.sample()
        for t in range(T):
            resultat[t] = li[t].value()
            
        return resultat

if __name__ == "__main__":
    delta_t = datetime.timedelta(minutes=30)
    time_horizon = datetime.timedelta(days=1)
    N = time_horizon // delta_t
    industrial_config = {
        'battery': {
            'capacity': 100,
            'efficiency': 0.95,
            'pmax': 25,
        },
        'building': {
            'site': 1,
        }
    }
    env = IndustrialEnv(industrial_config=industrial_config, nb_pdt=N)
    agent = IndustrialAgent(env)
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
