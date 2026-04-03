"""Print sample scenarios for each task difficulty."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ride_hailing_env.scenario_generator import ScenarioGenerator

WEATHER = ["clear", "rain", "storm"]
TRAFFIC = ["low", "medium", "heavy"]
LEVEL = ["low", "medium", "high"]
TIME = ["morning", "afternoon", "evening", "night"]
DAY = ["weekday", "weekend", "holiday"]

for task in ["easy", "medium", "hard"]:
    print("\n" + "=" * 70)
    print("TASK: %s" % task.upper())
    print("=" * 70)
    for i in range(5):
        gen = ScenarioGenerator(seed=42 + i)
        obs, hidden = gen.generate(task)
        overlap = hidden.rider_max_willingness - hidden.driver_min_willingness
        print("\n  Scenario %d (seed=%d):" % (i + 1, 42 + i))
        print("    Trip: %.1fkm, %.1fmin, ETA %.1fmin" % (obs.distance_km, obs.estimated_duration_min, obs.pickup_eta_min))
        print("    Context: weather=%s, traffic=%s, demand=%s, supply=%s" % (
            WEATHER[obs.weather_condition], TRAFFIC[obs.traffic_level],
            LEVEL[obs.demand_level], LEVEL[obs.supply_level]))
        print("    Time: %s, %s | Surge: %.2fx" % (TIME[obs.time_of_day], DAY[obs.day_type], obs.surge_multiplier))
        print("    Commission: %.0f%%, Op cost: $%.2f" % (obs.commission_rate * 100, obs.operational_cost))
        print("    Rider quote:  $%.2f  (max willing: $%.2f)" % (obs.rider_quoted_price, hidden.rider_max_willingness))
        print("    Driver quote: $%.2f  (min willing: $%.2f)" % (obs.driver_quoted_price, hidden.driver_min_willingness))
        print("    Visible gap:  $%.2f" % obs.price_gap)
        print("    Hidden overlap zone: $%.2f - $%.2f  (width=$%.2f)" % (
            hidden.driver_min_willingness, hidden.rider_max_willingness, overlap))
        print("    Patience init: rider=%.2f, driver=%.2f" % (obs.rider_patience, obs.driver_patience))
        print("    Patience decay: rider=%.4f, driver=%.4f" % (hidden.rider_patience_decay, hidden.driver_patience_decay))
        print("    Acceptance noise: rider=+/-$%.2f, driver=+/-$%.2f" % (hidden.rider_acceptance_noise, hidden.driver_acceptance_noise))
        print("    Max steps: %d" % obs.max_steps)
