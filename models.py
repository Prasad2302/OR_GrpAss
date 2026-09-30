"""Operations-research models for the HAL Defence Manufacturing project.

All inputs are academic planning assumptions (not HAL internal data).
"""
import itertools
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds, linear_sum_assignment

# ------------------------------------------------------------------ 1. LP: production mix
PRODUCTS = ["LCA Tejas Mk1A", "HTT-40", "ALH Dhruv", "LCH Prachand"]
LP_DEMAND = [16, 12, 15, 12]
LP_CONTRIB = [180, 85, 120, 150]                      # ₹ lakh / unit
RESOURCES = ["Skilled labour hrs", "Machine hrs", "Assembly hrs", "Testing hrs", "Critical components"]
LP_A = np.array([[1800, 1100, 1350, 1600],
                 [950, 600, 700, 820],
                 [1200, 750, 900, 1050],
                 [180, 110, 130, 150],
                 [1, 0.7, 0.8, 0.9]], float)
LP_AVAIL = [65000, 36000, 47000, 7000, 42]


def solve_lp(avail=LP_AVAIL):
    r = linprog(-np.array(LP_CONTRIB), A_ub=LP_A, b_ub=avail,
                bounds=list(zip([0] * 4, LP_DEMAND)), method="highs")
    return dict(x=r.x, obj=-r.fun, used=LP_A @ r.x, avail=np.array(avail, float),
                shadow=-r.ineqlin.marginals)


# ------------------------------------------------------------------ 2. Goal programming
GP_GOALS = [("Total production (units)", ">=", 40), ("Total contribution (₹ lakh)", ">=", 6000),
            ("Labour usage index", "<=", 300), ("Machine usage index", "<=", 220),
            ("Material usage index", "<=", 330)]
GP_MAX = [20, 18, 16, 14]
GP_WEIGHTS = [5, 4, 3, 2, 1]                          # priority 1 = highest weight
GP_COEF = np.vstack([np.ones(4), LP_CONTRIB, LP_A[0] / 280, LP_A[1] / 175, LP_A[2] / 143])


def solve_gp(targets):
    n = len(GP_GOALS)
    ge = np.array([g[1] == ">=" for g in GP_GOALS])
    cost = np.zeros(4 + 2 * n)
    cost[4:4 + n] = np.where(ge, GP_WEIGHTS, 0)       # penalise shortfall d- on >= goals
    cost[4 + n:] = np.where(ge, 0, GP_WEIGHTS)        # penalise excess d+ on <= goals
    cost[:4] = -1e-6 * np.array(LP_CONTRIB)           # tiny tie-break: prefer higher contribution
    A_eq = np.hstack([GP_COEF, np.eye(n), -np.eye(n)])
    r = linprog(cost, A_eq=A_eq, b_eq=targets,
                bounds=[(0, m) for m in GP_MAX] + [(0, None)] * (2 * n), method="highs")
    x, dm, dp = r.x[:4], r.x[4:4 + n], r.x[4 + n:]
    dev = np.where(ge, dm, dp)
    return dict(x=x, achieved=GP_COEF @ x, dev=dev, total=float(np.dot(dev, GP_WEIGHTS)))


# ------------------------------------------------------------------ 3. Integer programming
SUPPLIERS = ["AeroTech Components", "Bharat Aero Systems", "Precision Forge", "Deccan Aero Structures",
             "Composite India", "HydraMotion", "AeroFab Gujarat", "Global Aero Parts"]
IP_CAT = ["Avionics", "Avionics", "Engine Components", "Aero-structures", "Composite Structures",
          "Hydraulic Systems", "Aero-structures", "Engine Components"]
IP_CAP = [35, 30, 40, 45, 30, 25, 35, 30]
IP_PC = [32, 34, 28, 24, 21, 18, 23, 31]              # ₹ lakh / unit
IP_FX = [180, 150, 120, 100, 90, 80, 95, 130]         # fixed contract cost
IP_MOQ = [8, 6, 10, 12, 8, 6, 10, 8]
IP_DOM = [1, 1, 1, 1, 1, 1, 1, 0]
IP_DEMAND = {"Avionics": 40, "Engine Components": 35, "Aero-structures": 50,
             "Composite Structures": 20, "Hydraulic Systems": 15}


def solve_ip(max_suppliers=7, domestic_share=0.8):
    n = 8
    c = np.array(IP_PC + IP_FX, float)
    rows, lo, hi = [], [], []
    for cat, d in IP_DEMAND.items():
        rows.append([1.0 if IP_CAT[j] == cat else 0 for j in range(n)] + [0] * n); lo.append(d); hi.append(np.inf)
    for j in range(n):
        r = [0] * (2 * n); r[j] = 1; r[n + j] = -IP_CAP[j]; rows.append(r); lo.append(-np.inf); hi.append(0)
        r = [0] * (2 * n); r[j] = 1; r[n + j] = -IP_MOQ[j]; rows.append(r); lo.append(0); hi.append(np.inf)
    rows.append([0] * n + [1] * n); lo.append(0); hi.append(max_suppliers)
    rows.append(list(IP_DOM) + [0] * n); lo.append(domestic_share * sum(IP_DEMAND.values())); hi.append(np.inf)
    r = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=[1] * (2 * n),
             bounds=Bounds(0, [max(IP_CAP)] * n + [1] * n))
    if r.status != 0:
        return None
    q, y = np.round(r.x[:n]), np.round(r.x[n:])
    return dict(q=q, y=y, cost=float(r.fun), var=q * IP_PC, fix=y * IP_FX)


# ------------------------------------------------------------------ 4. Transportation
ORIGINS = ["Bengaluru", "Hyderabad", "Pune / Maharashtra", "Vadodara / Gujarat", "Chennai / Tamil Nadu"]
DESTS = ["HAL Bengaluru", "HAL Hyderabad", "HAL Nashik", "HAL Tumakuru"]
TR_COST = np.array([[4, 12, 15, 7], [5, 3, 14, 9], [8, 11, 4, 10], [13, 12, 9, 14], [10, 8, 12, 11]], float)
TR_SUPPLY = [120, 90, 100, 80, 90]
TR_DEMAND = [100, 85, 70, 95]


def solve_transport():
    a_ub, a_eq = np.zeros((5, 20)), np.zeros((4, 20))
    for i in range(5):
        for j in range(4):
            a_ub[i, i * 4 + j] = 1; a_eq[j, i * 4 + j] = 1
    r = linprog(TR_COST.flatten(), A_ub=a_ub, b_ub=TR_SUPPLY, A_eq=a_eq, b_eq=TR_DEMAND, method="highs")
    return dict(T=r.x.reshape(5, 4), cost=r.fun / 100)      # cost unit: ₹ lakh per 100 units


# ------------------------------------------------------------------ 5. Assignment
ENGINEERS = [f"Engineer E{i}" for i in range(1, 9)]
JOBS = ["LCA Avionics", "LCA Structural", "ALH Engine", "ALH Avionics",
        "LCH Systems", "LCH Airframe", "SU-30 Engine", "Do-228 MRO"]
AS_TIME = np.array([[18, 24, 30, 16, 28, 35, 40, 26], [22, 16, 28, 25, 30, 20, 38, 24],
                    [34, 30, 18, 32, 24, 28, 20, 35], [20, 26, 32, 18, 22, 30, 35, 20],
                    [28, 32, 26, 24, 18, 20, 30, 28], [30, 18, 25, 28, 20, 16, 32, 22],
                    [38, 34, 20, 36, 28, 30, 16, 32], [24, 22, 34, 20, 26, 24, 32, 18]])


def solve_assignment():
    r, c = linear_sum_assignment(AS_TIME)
    return dict(rows=r, cols=c, total=int(AS_TIME[r, c].sum()))


# ------------------------------------------------------------------ 6. Nonlinear programming
def nlp_cost(u):
    return 900 + 1800 * np.asarray(u) + 2400 * np.asarray(u) ** 2      # ₹ crore


def solve_nlp(min_production=75, u_min=0.6, u_max=1.0):
    u = max(u_min, min_production / 100)
    return dict(u=u, cost=float(nlp_cost(u)), feasible=u <= u_max)


# ------------------------------------------------------------------ 7. TSP (exact)
NODES = ["Bengaluru Base", "Tumakuru", "Hyderabad", "Korwa", "Lucknow", "Kanpur", "Nashik", "Koraput"]
NODE_LATLON = [(12.97, 77.59), (13.34, 77.10), (17.39, 78.49), (26.43, 81.65),
               (26.85, 80.95), (26.45, 80.35), (19.99, 73.79), (18.81, 82.71)]   # approximate
TSP_D = np.array([[0, 75, 575, 1600, 1850, 1750, 1050, 1250], [75, 0, 650, 1650, 1925, 1825, 975, 1325],
                  [575, 650, 0, 1350, 1250, 1180, 1450, 900], [1600, 1650, 1350, 0, 350, 270, 1050, 850],
                  [1850, 1925, 1250, 350, 0, 95, 1050, 1250], [1750, 1825, 1180, 270, 95, 0, 950, 1150],
                  [1050, 975, 1450, 1050, 1050, 950, 0, 1450], [1250, 1325, 900, 850, 1250, 1150, 1450, 0]])


def solve_tsp():
    best = min(((0,) + p + (0,) for p in itertools.permutations(range(1, 8))),
               key=lambda t: sum(TSP_D[t[k], t[k + 1]] for k in range(8)))
    legs = [int(TSP_D[best[k], best[k + 1]]) for k in range(8)]
    return dict(route=list(best), legs=legs, total=sum(legs))
