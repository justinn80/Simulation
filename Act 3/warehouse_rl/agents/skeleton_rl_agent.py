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
        WEEK 2 STEP 1: Staffing decisions - students should improve this!
        
        Current problems:
        - Ignores queue length and workload
        - Random decisions regardless of profit
        - No consideration of employee efficiency
        """
        
        # TODO WEEK 2 STEP 1: Students should implement intelligent staffing logic
        # Current approach: Make random decisions based on "vibes"
        
        current_profit = financial_state[0]
        num_employees = np.sum(employee_info[:, 0] > 0)  # Count active employees
        
        # Terrible logic: Random decisions with slight bias
        # BUT: Hire managers more frequently so layout optimization can work
        has_manager = np.any(employee_info[:, 5] == 1)  # Check if we have a manager
        
        if np.random.random() < 0.3:  # 30% chance to hire
            if num_employees < 20:  # Don't go completely overboard
                return 1  # Hire worker
        elif np.random.random() < 0.1:  # 10% chance to fire
            if num_employees > 1:  # Don't fire everyone
                return 2  # Fire worker
        elif np.random.random() < 0.2:  # 20% chance to hire manager (increased from 5%)
            if not has_manager or num_employees > 10:  # Hire manager if we don't have one, or if we have lots of workers
                return 3  # Hire manager
        
        return 0  # No action
    
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
    # HOOK: adapt to however your env exposes new orders (list of item-id lists)
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

    # build cell-level weights 
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
        WEEK 2 STEP 2: Order assignment - students should improve this!
        
        Current problems:
        - Ignores employee locations
        - No consideration of order priority/value
        - Random assignments regardless of efficiency
        - Doesn't check if employees are actually available
        """
        
        # TODO WEEK 2 STEP 2: Students should implement intelligent order assignment
        # Current approach: Random assignments
        
        assignments = [0] * 20  # No assignments by default
        
        # Count how many employees we have (very naive)
        num_employees = int(np.sum(employee_info[:, 0] > 0))
        
        # Randomly assign first few orders to first few employees
        if num_employees > 0:
            for i in range(min(3, num_employees)):  # Only assign 3 orders max
                if np.random.random() < 0.6:  # 60% chance to assign
                    assignments[i] = (i % num_employees) + 1  # Random employee
        
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
        WEEK 2 STEP 1: Students implement intelligent staffing:
        - Hire when queue is growing
        - Fire when queue is empty for extended periods
        - Consider profit margins before hiring
        - Balance managers vs workers
        """
        pass
    
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
        WEEK 2 STEP 2: Students implement efficient order assignment:
        - Assign orders to closest available employees
        - Prioritize high-value or urgent orders
        - Consider employee current locations
        - Balance workload across employees
        """
        pass
    
    def learn_from_experience(self, state, action, reward, next_state, done):
        """
        WEEK 3 STEP 3: Students implement multi-objective optimization:
        - Performance trend analysis
        - Adaptive parameter tuning
        - Multi-objective trade-off handling
        - Robust optimization techniques
        """
        pass
