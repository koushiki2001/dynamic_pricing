# Reference — Dynamic Pricing Hackathon

Research and findings compiled during ideation and design phase. Use these as supporting material for the presentation.

## Files

| File | Contents |
|---|---|
| [01_current_pricing_systems.md](01_current_pricing_systems.md) | How Uber/Rapido price rides today, what ML stack they use, and the 7 dimensions current systems leave untouched |
| [02_multi_agent_architecture.md](02_multi_agent_architecture.md) | Two-LLM design (Platform LLM + Simulator LLM), agent objectives, cooperative-competitive dynamic, training schedule, expected emergent behaviors |
| [03_rl_training_approach.md](03_rl_training_approach.md) | Why GRPO over PPO, TRL + Unsloth stack, rollout function design, curriculum strategy, reward-as-verifier approach, model saving warnings |
| [04_episode_walkthroughs.md](04_episode_walkthroughs.md) | Concrete step-by-step episodes for easy / medium / hard difficulties showing what each LLM agent observes, reasons, and decides — with actual reward numbers |
| [05_behavioral_signals_deployment.md](05_behavioral_signals_deployment.md) | How simulation features (patience, mood, threshold) map to real-world observable signals, the feature assembly pipeline, calibration loop, A/B testing approach |
| [06_project_gap_analysis.md](06_project_gap_analysis.md) | Current project state vs hackathon requirements, what is complete, what is missing, feasibility table, and the strongest defensible claim for the presentation |

## Key Thesis (One Paragraph)

Current ride-hailing pricing systems optimize for market-level supply-demand balance using aggregate signals. They do not model individual user behavioral signals (urgency, patience, strategic testing). This project builds a multi-agent RL environment where a platform LLM learns to propose prices that close deals efficiently by inferring hidden user thresholds from behavioral signals — trained against a strategic simulator LLM that learns optimal bluffing behavior. The emergent equilibrium approximates real-world negotiation dynamics more accurately than surge-multiplier systems, and the trained platform model's reasoning translates directly to a behavioral pricing oracle for deployment.
