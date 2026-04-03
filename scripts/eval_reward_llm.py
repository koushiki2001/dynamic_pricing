"""Run the official grader on reward-guided LLM + baselines for comparison."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from baselines.evaluate_baselines import run_stateless_baseline
from baselines.reward_guided_llm_policy import RewardGuidedLLMPolicy
from baselines.midpoint_policy import midpoint_policy
from baselines.adaptive_policy import AdaptivePolicy
from ride_hailing_env.config import TASK_CONFIG, MAX_EFFICIENCY_BONUS
from ride_hailing_env.environment import DynamicPricingEnv
from ride_hailing_env.utils import normalize_revenue
from ride_hailing_env.tasks.graders import GRADER_WEIGHTS


def run_llm_baseline(task_name, policy_class, num_episodes, seed=None):
    """Like run_stateful_baseline but handles duplicate-price guard."""
    cfg = TASK_CONFIG[task_name]
    seed = seed or cfg["seed"]

    total_completed = 0
    total_cancelled = 0
    total_timed_out = 0
    total_profit = 0.0
    total_efficiency = 0.0
    completed_count = 0

    for ep in range(num_episodes):
        ep_seed = seed + ep
        env = DynamicPricingEnv(task_name=task_name, seed=ep_seed)
        obs = env.reset()
        done = False
        policy = policy_class()
        policy.reset()
        iters = 0

        while not done and iters < 50:
            iters += 1
            action = policy(obs.model_dump())
            result = env.step(action)
            # Handle duplicate price guard
            if result.info.get("duplicate_price"):
                price = action["payload"]["price"]
                action["payload"]["price"] = round(price + 1.0, 2)
                result = env.step(action)
            obs = result.observation
            done = result.done

        outcome = result.info["outcome"]
        if outcome["ride_completed"]:
            total_completed += 1
            completed_count += 1
            total_profit += outcome["platform_profit"]
            total_efficiency += cfg["max_steps"] / outcome["steps_taken"]
        elif outcome["timed_out"]:
            total_timed_out += 1
        else:
            total_cancelled += 1

    completion_rate = total_completed / num_episodes
    cancellation_rate = total_cancelled / num_episodes
    avg_efficiency = total_efficiency / completed_count if completed_count > 0 else 0.0
    avg_efficiency_norm = min(avg_efficiency / MAX_EFFICIENCY_BONUS, 1.0)
    avg_profit = total_profit / completed_count if completed_count > 0 else 0.0
    profit_norm = normalize_revenue(avg_profit)

    w = GRADER_WEIGHTS[task_name]
    score = (
        w["completion"] * completion_rate
        + w["efficiency"] * avg_efficiency_norm
        + w["profit"] * profit_norm
        + w["no_cancel"] * (1.0 - cancellation_rate)
    )
    score = max(0.0, min(1.0, score))

    return {
        "task": task_name,
        "score": round(score, 4),
        "completion_rate": round(completion_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "timeout_rate": round(total_timed_out / num_episodes, 4),
        "avg_profit": round(avg_profit, 4),
        "avg_efficiency": round(avg_efficiency, 4),
        "num_episodes": num_episodes,
    }


N_EPS = 15  # practical with LLM calls (~2s each)

tasks = ["easy", "medium", "hard"]

print("=" * 70)
print("REWARD-GUIDED LLM - OFFICIAL GRADER EVALUATION")
print(f"({N_EPS} episodes per task)")
print("=" * 70)

rg_scores = []
for task in tasks:
    t0 = time.time()
    r = run_llm_baseline(task, RewardGuidedLLMPolicy, num_episodes=N_EPS)
    elapsed = time.time() - t0
    rg_scores.append(r["score"])
    print(
        f"  {task:>8s}: score={r['score']:.4f}  "
        f"completion={r['completion_rate']:.2%}  "
        f"cancel={r['cancellation_rate']:.2%}  "
        f"timeout={r['timeout_rate']:.2%}  "
        f"profit=${r['avg_profit']:.2f}  "
        f"efficiency={r['avg_efficiency']:.2f}  "
        f"({elapsed:.1f}s)"
    )
print(f"  {'avg':>8s}: {sum(rg_scores)/len(rg_scores):.4f}")

print("\n--- Midpoint (comparison) ---")
mid_scores = []
for task in tasks:
    r = run_stateless_baseline(task, midpoint_policy, num_episodes=N_EPS)
    mid_scores.append(r["score"])
    print(
        f"  {task:>8s}: score={r['score']:.4f}  "
        f"completion={r['completion_rate']:.2%}  "
        f"cancel={r['cancellation_rate']:.2%}  "
        f"profit=${r['avg_profit']:.2f}"
    )
print(f"  {'avg':>8s}: {sum(mid_scores)/len(mid_scores):.4f}")

print("\n--- Adaptive (comparison) ---")
from baselines.evaluate_baselines import run_stateful_baseline
adp_scores = []
for task in tasks:
    r = run_stateful_baseline(task, AdaptivePolicy, num_episodes=N_EPS)
    adp_scores.append(r["score"])
    print(
        f"  {task:>8s}: score={r['score']:.4f}  "
        f"completion={r['completion_rate']:.2%}  "
        f"cancel={r['cancellation_rate']:.2%}  "
        f"profit=${r['avg_profit']:.2f}"
    )
print(f"  {'avg':>8s}: {sum(adp_scores)/len(adp_scores):.4f}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Midpoint:            {sum(mid_scores)/len(mid_scores):.4f}")
print(f"  Adaptive:            {sum(adp_scores)/len(adp_scores):.4f}")
print(f"  Reward-Guided LLM:   {sum(rg_scores)/len(rg_scores):.4f}")
print("=" * 70)
