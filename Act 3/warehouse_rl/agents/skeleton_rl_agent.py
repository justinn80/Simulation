#!/usr/bin/env python3
"""
Skeleton Optimization Agent - Template for students to implement optimization algorithms

IMPORTANT: This is NOT an RL agent - it's a template for 
students to implement their own optimization algorithms (greedy, Hungarian, etc.)

This agent currently makes terrible decisions:
- Random staffing decisions
- No layout optimization 
- Ignores order priorities
- No intelligent assignment logic

Students should replace the naive methods with proper optimization algorithms.
"""

import numpy as np
from typing import Dict, Optional
from .standardized_agents import BaselineAgent

class SkeletonOptimizationAgent(BaselineAgent):
    """
    Template for students to implement their own optimization algorithms.
    
    Students should replace the naive methods with:
    - Economic models for staffing decisions
    - Hungarian algorithm for order assignment  
    - Greedy search for layout optimization
    - Multi-objective optimization techniques
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # Enable unlimited hiring for true economic optimization
        self.env._unlimited_hiring = True
        
        # Students can add their own state tracking here
        self.performance_metrics = []
        self.decision_history = []
        
        # Students can add algorithm parameters here
        # Example: Hungarian algorithm matrices, greedy search parameters, etc.
        self.staffing_parameters = {
            'hire_threshold_ratio': 3.0,
            'fire_threshold_ratio': 2.0,
            'profit_threshold': 0
        }
        
        # Students can implement adaptive parameters that change based on performance
        self.adaptive_optimization_enabled = False
        
    def reset(self):
        """Reset agent state - students should expand this"""
        self.action_history = []
        self.reward_history = []
        # TODO: Reset any neural network states, replay buffers, etc.
    
    def get_action(self, observation: Dict) -> Dict:
        """
        Generate action based on observation.
        Current implementation is intentionally terrible - students should improve!
        """
        
        # Extract basic info from observation
        current_timestep = observation['time'][0]
        financial_state = observation['financial']  # [profit, revenue, costs, burn_rate]
        queue_info = observation['order_queue']
        employee_info = observation['employees']
        
        action = {
            'staffing_action': self._get_naive_staffing_action(financial_state, employee_info),
            'layout_swap': self._get_naive_layout_action(current_timestep),
            'order_assignments': self._get_naive_order_assignments(queue_info, employee_info)
        }
        
        # TODO: Students should implement proper action recording for optimization
        self.action_history.append(action.copy())
        
        return action
    
    def _get_naive_staffing_action(self, financial_state, employee_info) -> int:
    """
    Workload- and profit-aware staffing with hysteresis and cooldown.
    Returns 0 = none, 1 = hire worker, 2 = fire worker, 3 = hire manager.
    """
    # tunables 
    MAX_STAFF, MIN_WORKERS = 20, 1
    WORKERS_PER_MANAGER = 8
    COOLDOWN = 15
    EMA = 0.1
    HIRE_QPW, FIRE_QPW = 2.0, 0.5       # queued orders per worker thresholds
    HIRE_UTIL, FIRE_UTIL = 0.85, 0.40   # busy fraction thresholds
    E_EXISTS, E_BUSY, E_MANAGER = 0, 3, 5

    # ---- state ----
    if not hasattr(self, "_staff"):
        self._staff = dict(t=0, last=-COOLDOWN, profit_ema=None, trend=0.0)
    s = self._staff
    s["t"] += 1

    profit = float(financial_state[0])
    prev = s["profit_ema"]
    s["profit_ema"] = profit if prev is None else prev + EMA * (profit - prev)
    s["trend"] = 0.0 if prev is None else (1 - EMA) * s["trend"] + EMA * (s["profit_ema"] - prev)

    if s["t"] - s["last"] < COOLDOWN:
        return 0

    # workforce snapshot (vectorized) 
    active = employee_info[:, E_EXISTS] > 0
    is_mgr = active & (employee_info[:, E_MANAGER] == 1)
    n_total = int(active.sum())
    n_mgr = int(is_mgr.sum())
    n_workers = n_total - n_mgr
    busy = active & ~is_mgr & (employee_info[:, E_BUSY] > 0)
    util = busy.sum() / n_workers if n_workers > 0 else 1.0
    queue = self._get_staffing_queue_length()           # HOOK
    qpw = queue / max(n_workers, 1) if queue is not None else (util * HIRE_QPW)

    healthy = s["profit_ema"] >= 0 or s["trend"] > 0
    losing_badly = s["profit_ema"] < 0 and s["trend"] < 0

    action = 0
    if n_workers < MIN_WORKERS and n_total < MAX_STAFF:
        action = 1                                       # never runs with no workers
    elif n_mgr == 0 and healthy and n_total < MAX_STAFF:
        action = 3                                       # needs a manager for layout
    elif n_total < MAX_STAFF and (qpw > HIRE_QPW or util > HIRE_UTIL):
        # hire into demand if profit is falling, only for a severe backlog
        if not losing_badly or qpw > 2 * HIRE_QPW:
            needs_mgr = n_workers >= WORKERS_PER_MANAGER * (n_mgr + 1) and healthy
            action = 3 if needs_mgr else 1
    elif n_workers > MIN_WORKERS and qpw < FIRE_QPW and util < FIRE_UTIL:
        action = 2                                       # idle capacity is pure cost
    elif n_workers > MIN_WORKERS and losing_badly and util < HIRE_UTIL:
        action = 2                                       # shrink to stop the bleeding

    if action:
        s["last"] = s["t"]
    return action

    def _get_staffing_queue_length(self):
    """return the current order queue length, or None if unavailable."""
    q = getattr(self.env, "queue_length", None)
    if q is None:
        qi = getattr(self.env, "queue_info", None)
        if qi is not None:
            q = int((np.asarray(qi)[:, 0] > 0).sum())
    return q
    
    def _get_naive_layout_action(self, current_timestep) -> list:
    """
    Layout optimization: greedy best swap that minimizes
        sum_{i<j} C[i,j] * dist(pos_i, pos_j)  +  sum_i F[i] * dist(pos_i, depot)
    where C = decayed item co-occurrence and F = decayed item demand.
    Evaluates all cell-pair swaps in one vectorized pass.
    Checks worth of each swap considered.
    """
    NO_OP = [0, 0]
    CHECK_EVERY, COOLDOWN = 25, 100
    DECAY, MIN_REL_GAIN = 0.98, 0.02

    env = self.env
    W, H = env.grid_width, env.grid_height
    n_cells = W * H

    # one-time setup (cached) 
    if not hasattr(self, "_lay"):
        xs, ys = np.divmod(np.arange(n_cells), W)[::-1]  # x = idx % W, y = idx // W
        coords = np.stack([xs, ys], axis=1)
        D = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1).astype(float)  
        dx, dy = getattr(env, "delivery_point", (0, 0))
        Dd = np.abs(xs - dx) + np.abs(ys - dy)
        n_items = env.num_items
        self._lay = dict(D=D, Dd=Dd.astype(float),
                         C=np.zeros((n_items, n_items)), F=np.zeros(n_items),
                         last_swap=-COOLDOWN)

    s = self._lay
    C, F, D, Dd = s["C"], s["F"], s["D"], s["Dd"]

    # incremental demand statistics 
    new_orders = self._get_orders_since_last_call()
    if new_orders:
        C *= DECAY
        F *= DECAY
        for order in new_orders:
            items = np.unique(order)
            F[items] += 1.0
            if len(items) > 1:
                C[np.ix_(items, items)] += 1.0
        np.fill_diagonal(C, 0.0)

    # throttle
    if current_timestep % CHECK_EVERY or current_timestep - s["last_swap"] < COOLDOWN:
        return NO_OP
    if F.sum() == 0:
        return NO_OP

    # build cell level weights 
    layout = np.asarray(env.layout).ravel()
    occupied = layout >= 0
    ids = np.where(occupied, layout, 0)

    Fc = np.where(occupied, F[ids], 0.0)                              
    Cc = C[np.ix_(ids, ids)] * np.outer(occupied, occupied)          

    G = Cc @ D + np.outer(Fc, Dd)
    own = np.diag(G)

    delta = G + G.T - own[:, None] - own[None, :] + 2.0 * Cc * D
    np.fill_diagonal(delta, 0.0)
    delta[np.triu_indices(n_cells)] = 0.0          # each unordered pair once

    # pick best swap, only if worth it 
    flat = int(np.argmin(delta))
    a, b = divmod(flat, n_cells)
    gain = -delta[a, b]
    total_cost = 0.5 * own.sum()                    
    if gain <= MIN_REL_GAIN * max(total_cost, 1e-9) or (not occupied[a] and not occupied[b]):
        return NO_OP

    s["last_swap"] = current_timestep
    return [int(a), int(b)]
    
    def _get_naive_order_assignments(self, queue_info, employee_info) -> list:
    """
    Optimal order assignment: maximizes sum of
        value * (1 + urgency) / (1 + distance(employee -> pickup))
    over all (order, free worker) pairs using the Hungarian algorithm.
    """
    # column layout 
    E_EXISTS, E_X, E_Y, E_BUSY, E_MANAGER = 0, 1, 2, 3, 5
    Q_EXISTS, Q_X, Q_Y, Q_VALUE, Q_PRIORITY = 0, 1, 2, 3, 4
    N_SLOTS = 20

    assignments = [0] * N_SLOTS

    queue_info = np.asarray(queue_info, dtype=float)
    employee_info = np.asarray(employee_info, dtype=float)

    # valid orders and available workers 
    order_idx = np.flatnonzero(queue_info[:N_SLOTS, Q_EXISTS] > 0)
    avail = employee_info[:, E_EXISTS] > 0
    if employee_info.shape[1] > E_BUSY:
        avail &= employee_info[:, E_BUSY] <= 0
    if employee_info.shape[1] > E_MANAGER:
        avail &= employee_info[:, E_MANAGER] != 1      # so managers don't fulfil orders
    emp_idx = np.flatnonzero(avail)

    if order_idx.size == 0 or emp_idx.size == 0:
        return assignments

    Q = queue_info[order_idx]
    E = employee_info[emp_idx]

    # order score: value * urgency (degrades gracefully if columns are absent) 
    ncols = Q.shape[1]
    value = Q[:, Q_VALUE] if ncols > Q_VALUE else np.ones(len(Q))
    urgency = 1.0 + (np.maximum(Q[:, Q_PRIORITY], 0.0) if ncols > Q_PRIORITY else 0.0)
    score = np.maximum(value, 1e-6) * urgency

    # distance matrix (orders x workers), Manhattan 
    dist = (np.abs(Q[:, None, Q_X] - E[None, :, E_X]) +
            np.abs(Q[:, None, Q_Y] - E[None, :, E_Y]))

    benefit = score[:, None] / (1.0 + dist)

    # optimal matching (rectangular is fine) 
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(benefit, maximize=True)
    except ImportError:
        # greedy fallback: repeatedly take the best remaining pair
        b = benefit.copy()
        rows, cols = [], []
        for _ in range(min(b.shape)):
            r, c = np.unravel_index(np.argmax(b), b.shape)
            rows.append(r); cols.append(c)
            b[r, :] = -np.inf
            b[:, c] = -np.inf

    for r, c in zip(rows, cols):
        assignments[int(order_idx[r])] = int(emp_idx[c]) + 1   # 1-based employee id

    return assignments
    
    def record_reward(self, reward: float):
        """
        WEEK 3 STEP 1: Reward tracking and learning - students should expand this
        
        TODO WEEK 3 STEP 1: Students should implement:
        - Proper reward tracking
        - Performance analysis
        - Adaptive parameter adjustment
        - Multi-objective optimization
        """
        self.reward_history.append(reward)
        
        # Skeleton optimization - doesn't actually learn anything useful
        if len(self.reward_history) > 10:
            # "Update" weights randomly (this doesn't actually improve performance)
            if reward > 0:
                self.staffing_weights += np.random.randn(4) * 0.01
                self.layout_weights += np.random.randn(3) * 0.01
    
    def should_update_policy(self) -> bool:
        """
        WEEK 3 STEP 2: Policy updates and adaptation - students should improve this
        
        TODO WEEK 3 STEP 2: Students should implement proper update schedules
        """
        return len(self.action_history) % 50 == 0  # Arbitrary update frequency
    
    def get_performance_metrics(self) -> Dict:
        """
        Get agent performance metrics for analysis
        Students can use this to debug their improvements
        """
        if not self.reward_history:
            return {"avg_reward": 0, "total_actions": 0}
        
        return {
            "avg_reward": np.mean(self.reward_history[-100:]),  # Last 100 rewards
            "total_actions": len(self.action_history),
            "exploration_rate": self.exploration_rate,
            "recent_performance": np.mean(self.reward_history[-10:]) if len(self.reward_history) >= 10 else 0
        }


def create_skeleton_optimization_agent(env) -> SkeletonOptimizationAgent:
    """Factory function to create skeleton Optimization agent"""
    return SkeletonOptimizationAgent(env)


# TODO: Students should implement these advanced components:

class StudentOptimizationAgent(SkeletonOptimizationAgent):
    """
    Template for students to implement their improved Optimization agent
    
    Suggested improvements:
    1. Replace random staffing with demand-based hiring
    2. Implement proper layout optimization using item frequencies
    3. Add distance-based order assignment
    4. Implement basic Q-optimization or policy gradients
    5. Add proper exploration vs exploitation balance
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.name = "StudentOptimization"
        
        # TODO: Students implement these
        # self.q_table = {}
        # self.policy_network = SimpleNeuralNetwork()
        # self.experience_buffer = []
        # self.target_network = None
        
    def _get_improved_staffing_action(self, financial_state, employee_info, queue_info):
    """
    Queue-trend, profit, and manager-ratio aware staffing.
    Returns 0 = none, 1 = hire worker, 2 = fire worker, 3 = hire manager.
    """
    # ---- tunables ----
    MAX_STAFF, MIN_WORKERS = 20, 1
    WORKERS_PER_MANAGER = 8
    COOLDOWN = 15
    FAST, SLOW, PROFIT_EMA = 0.3, 0.05, 0.1
    HIRE_QPW, SEVERE_QPW = 1.5, 4.0       # queued orders per worker
    GROWTH_MARGIN = 0.5                   # fast avg must exceed slow avg by this
    EMPTY_STEPS = 40                      # empty-queue streak before firing
    MIN_PROFIT = 0.0                      # smoothed profit required to hire
    E_EXISTS, E_BUSY, E_MANAGER = 0, 3, 5

    # ---- state ----
    if not hasattr(self, "_staff2"):
        self._staff2 = dict(t=0, last=-COOLDOWN, q_fast=0.0, q_slow=0.0,
                            empty=0, profit=None, trend=0.0)
    s = self._staff2
    s["t"] += 1

    # ---- queue statistics ----
    queue = int((np.asarray(queue_info)[:, 0] > 0).sum())
    s["q_fast"] += FAST * (queue - s["q_fast"])
    s["q_slow"] += SLOW * (queue - s["q_slow"])
    s["empty"] = s["empty"] + 1 if queue == 0 else 0
    growing = s["q_fast"] > s["q_slow"] + GROWTH_MARGIN

    # ---- profit statistics ----
    profit = float(financial_state[0])
    prev = s["profit"]
    s["profit"] = profit if prev is None else prev + PROFIT_EMA * (profit - prev)
    s["trend"] = 0.0 if prev is None else (1 - PROFIT_EMA) * s["trend"] + PROFIT_EMA * (s["profit"] - prev)
    can_afford = s["profit"] >= MIN_PROFIT
    losing = s["profit"] < 0 and s["trend"] < 0

    if s["t"] - s["last"] < COOLDOWN:
        return 0

    # ---- workforce snapshot (vectorized)
    active = employee_info[:, E_EXISTS] > 0
    is_mgr = active & (employee_info[:, E_MANAGER] == 1)
    n_total = int(active.sum())
    n_mgr = int(is_mgr.sum())
    n_workers = n_total - n_mgr
    n_busy = int((active & ~is_mgr & (employee_info[:, E_BUSY] > 0)).sum())
    n_idle = n_workers - n_busy
    qpw = queue / max(n_workers, 1)
    target_mgr = max(1, -(-n_workers // WORKERS_PER_MANAGER))   # ceil division
    room = n_total < MAX_STAFF

    action = 0
    if n_workers < MIN_WORKERS and room:
        action = 1                                               # never run with no workers
    elif n_mgr == 0 and can_afford and room:
        action = 3                                               # need a manager for layout
    elif room and n_idle < queue and (
            (growing and qpw > HIRE_QPW and can_afford)
            or (qpw > SEVERE_QPW and not losing)):
        action = 3 if n_mgr < target_mgr and can_afford else 1   # hire into demand
    elif n_workers > MIN_WORKERS and s["empty"] >= (EMPTY_STEPS // 2 if losing else EMPTY_STEPS):
        action = 2                                               # idle capacity is pure cost

    if action:
        s["last"] = s["t"]
        s["empty"] = 0 if action == 2 else s["empty"]            # re-earn the next firing
    return action
    
    def _get_improved_layout_action(self, observation):
    """
    Smart layout optimization. Greedy best-swap on
        cost = sum_{i<j} C[i,j]*dist(i,j) + sum_i F[i]*dist(i, delivery)
    with C = decayed co-occurrence, F = decayed demand.
    Gated by queue size, cooldown, and an adaptive threshold that learns
    from how well past swaps actually worked.
    """
    from collections import deque

    NO_OP = [0, 0]
    CHECK_EVERY, COOLDOWN = 20, 80
    MAX_QUEUE = 8                 
    DECAY = 0.98
    EVAL_WINDOW = 60             
    TABU_STEPS = 300              
    GAIN_MIN, GAIN_MAX = 0.005, 0.10

    env = self.env
    W, H = env.grid_width, env.grid_height
    n_cells = W * H

    #  one-time setup 
    if not hasattr(self, "_il"):
        idx = np.arange(n_cells)
        xs, ys = idx % W, idx // W
        coords = np.stack([xs, ys], axis=1)
        D = np.abs(coords[:, None, :] - coords[None, :, :]).sum(-1).astype(float)
        dx, dy = getattr(env, "delivery_point", (0, 0))
        Dd = (np.abs(xs - dx) + np.abs(ys - dy)).astype(float)
        n_items = env.num_items
        self._il = dict(
            D=D, Dd=Dd, t=0,
            C=np.zeros((n_items, n_items)), F=np.zeros(n_items),
            metric=deque(maxlen=400),            
            last_swap_t=-COOLDOWN,
            pending=None,                       
            tabu={},                             
            min_gain=0.02,
            history=[],                          
        )
    s = self._il
    s["t"] += 1
    t = s["t"]
    C, F, D, Dd = s["C"], s["F"], s["D"], s["Dd"]

    queue_len, new_orders, metric = self._parse_layout_observation(observation)

    # update demand statistics 
    if new_orders:
        C *= DECAY
        F *= DECAY
        for order in new_orders:
            items = np.unique(order)
            F[items] += 1.0
            if len(items) > 1:
                C[np.ix_(items, items)] += 1.0
        np.fill_diagonal(C, 0.0)

    if metric is not None:
        s["metric"].append((t, float(metric)))

    # evaluate the last swap's effectiveness 
    p = s["pending"]
    if p is not None and t - p["t"] >= EVAL_WINDOW and metric is not None:
        before = [v for (tt, v) in s["metric"] if p["t"] - EVAL_WINDOW <= tt < p["t"]]
        after = [v for (tt, v) in s["metric"] if tt >= p["t"]]
        if before and after:
            b, a = np.mean(before), np.mean(after)
            realized = (b - a) / max(abs(b), 1e-9)       
            s["history"].append((p["t"], p["gain"], realized))
            # adapt aggressiveness: effective -> lower threshold, harmful -> raise it
            s["min_gain"] = float(np.clip(
                s["min_gain"] * (0.85 if realized > 0 else 1.4), GAIN_MIN, GAIN_MAX))
        s["pending"] = None
    elif p is not None and t - p["t"] >= EVAL_WINDOW:
        s["pending"] = None                              

    # gating 
    if (t % CHECK_EVERY or t - s["last_swap_t"] < COOLDOWN
            or queue_len > MAX_QUEUE or F.sum() == 0):
        return NO_OP

    # cell-level weights 
    layout = np.asarray(env.layout).ravel()
    occ = layout >= 0
    ids = np.where(occ, layout, 0)
    Fc = np.where(occ, F[ids], 0.0)
    Cc = C[np.ix_(ids, ids)] * np.outer(occ, occ)

    # G[a, q]: cost of the entity at cell a if it were relocated to cell q
    G = Cc @ D + np.outer(Fc, Dd)
    own = np.diag(G)

    # exact total cost change for swapping cells a and b (vectorized over all pairs)
    delta = G + G.T - own[:, None] - own[None, :] + 2.0 * Cc * D
    delta = np.triu(delta, 1)                            

    # block recently-made swaps (tabu)
    for key in [k for k, exp in s["tabu"].items() if exp <= t]:
        del s["tabu"][key]
    for key in s["tabu"]:
        a_, b_ = sorted(key)
        delta[a_, b_] = 0.0

    # pick best swap if it clears the adaptive threshold 
    a, b = divmod(int(np.argmin(delta)), n_cells)
    gain = -delta[a, b]
    total_cost = 0.5 * float((Cc * D).sum()) + float((Fc * Dd).sum())
    if gain <= s["min_gain"] * max(total_cost, 1e-9) or not (occ[a] or occ[b]):
        return NO_OP

    s["last_swap_t"] = t
    s["pending"] = dict(t=t, gain=gain / max(total_cost, 1e-9))
    s["tabu"][frozenset((a, b))] = t + TABU_STEPS
    return [int(a), int(b)]

    def _parse_layout_observation(self, observation):
    """
    HOOK: return (queue_length, new_orders, performance_metric).
      new_orders: list of item-id lists completed/received since last call
      performance_metric: lower-is-better scalar (e.g. avg fulfilment time) or None
    """
    env = self.env
    if isinstance(observation, dict):
        queue_len = int(observation.get("queue_length",
                        np.sum(np.asarray(observation.get("queue_info", [[0]]))[:, 0] > 0)))
        metric = observation.get("avg_fulfillment_time")
    else:
        queue_len = int(getattr(env, "queue_length", 0))
        metric = getattr(env, "avg_fulfillment_time", None)

    orders = getattr(env, "completed_orders", [])
    start = getattr(self, "_orders_seen", 0)
    self._orders_seen = len(orders)
    return queue_len, orders[start:], metric
    
    def _get_improved_order_assignments(self, queue_info, employee_info):
    """
    Optimal, workload-balanced order assignment. Maximizes
        value * (1 + urgency)^2 / (1 + dist(worker -> pickup)) / (1 + BALANCE * load[worker])
    over all (order, available worker) pairs via the Hungarian algorithm.
    """
    # column layout 
    E_EXISTS, E_X, E_Y, E_BUSY, E_MANAGER = 0, 1, 2, 3, 5
    Q_EXISTS, Q_X, Q_Y, Q_VALUE, Q_PRIORITY = 0, 1, 2, 3, 4
    N_SLOTS = 20
    URGENCY_POWER = 2.0     # higher = urgent orders dominate more
    BALANCE = 0.3           # higher = spread work more evenly
    LOAD_DECAY = 0.95       # per-call decay of the assignment counter

    assignments = [0] * N_SLOTS

    queue_info = np.asarray(queue_info, dtype=float)
    employee_info = np.asarray(employee_info, dtype=float)
    n_emp = employee_info.shape[0]

    # workload tracker (resized if the workforce changes) 
    load = getattr(self, "_assign_load", None)
    if load is None or load.shape[0] != n_emp:
        load = np.zeros(n_emp)
    load *= LOAD_DECAY

    # valid orders and available workers 
    order_idx = np.flatnonzero(queue_info[:N_SLOTS, Q_EXISTS] > 0)
    avail = employee_info[:, E_EXISTS] > 0
    if employee_info.shape[1] > E_BUSY:
        avail &= employee_info[:, E_BUSY] <= 0
    if employee_info.shape[1] > E_MANAGER:
        avail &= employee_info[:, E_MANAGER] != 1      # managers don't fulfil orders
    emp_idx = np.flatnonzero(avail)

    if order_idx.size == 0 or emp_idx.size == 0:
        self._assign_load = load
        return assignments

    Q = queue_info[order_idx]
    E = employee_info[emp_idx]

    # order score: value * urgency (degrades gracefully if columns are absent) 
    ncols = Q.shape[1]
    value = np.maximum(Q[:, Q_VALUE], 1e-6) if ncols > Q_VALUE else np.ones(len(Q))
    urgency = (1.0 + np.maximum(Q[:, Q_PRIORITY], 0.0)) ** URGENCY_POWER if ncols > Q_PRIORITY \
        else np.ones(len(Q))
    score = value * urgency

    # benefit matrix (orders x workers) 
    dist = (np.abs(Q[:, None, Q_X] - E[None, :, E_X]) +
            np.abs(Q[:, None, Q_Y] - E[None, :, E_Y]))
    benefit = score[:, None] / (1.0 + dist) / (1.0 + BALANCE * load[emp_idx])[None, :]

    # optimal matching (rectangular is fine) 
    try:
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(benefit, maximize=True)
    except ImportError:
        # greedy fallback: repeatedly take the best remaining pair
        b = benefit.copy()
        rows, cols = [], []
        for _ in range(min(b.shape)):
            r, c = np.unravel_index(np.argmax(b), b.shape)
            rows.append(r); cols.append(c)
            b[r, :] = -np.inf
            b[:, c] = -np.inf

    for r, c in zip(rows, cols):
        e = int(emp_idx[c])
        assignments[int(order_idx[r])] = e + 1          # 1-based employee id
        load[e] += 1.0

    self._assign_load = load
    return assignments
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        WEEK 3 STEP 3: Students implement multi-objective optimization:
        - Performance trend analysis
        - Adaptive parameter tuning
        - Multi-objective trade-off handling
        - Robust optimization techniques
        """
        pass
