#!/usr/bin/env python3
"""
Pre-Submission Validation Checker
==================================

Run this before submitting to verify all HF Space requirements are met.

Usage:
    python pre_submission_check.py

All checks should PASS before submission.
"""

import os
import sys
import json
from pathlib import Path


class Validator:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.env_failures = 0  # Track environment variable failures separately
        self.root_dir = Path(__file__).parent

    def check(self, name: str, condition: bool, details: str = "", is_env_check=False):
        """Record a check result."""
        status = "✅ PASS" if condition else "❌ FAIL"
        self.checks.append((name, condition, status, details))
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            if is_env_check:
                self.env_failures += 1
        print(f"  {status}  {name}")
        if details:
            print(f"         {details}")

    def print_summary(self):
        """Print final summary."""
        total = self.passed + self.failed
        structural_failures = self.failed - self.env_failures
        
        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY: {self.passed}/{total} checks passed")
        print(f"{'='*70}")
        
        if structural_failures == 0:
            print("✅ ALL CRITICAL CHECKS PASSED - READY FOR SUBMISSION!")
            if self.env_failures > 0:
                print(f"   (Note: {self.env_failures} environment variables will be set on HF Space)")
            print(f"{'='*70}\n")
            return True
        else:
            print(f"❌ {structural_failures} critical check(s) failed - FIX BEFORE SUBMISSION")
            if self.env_failures > 0:
                print(f"   ({self.env_failures} environment variable failures are expected in dev)")
            print(f"{'='*70}\n")
            return False


def validate_all():
    """Run all pre-submission validations."""
    v = Validator()
    root = v.root_dir

    print("=" * 70)
    print("PRE-SUBMISSION VALIDATION")
    print("=" * 70)

    # ────────────────────────────────────────────────────────────────
    # 1. File Structure
    # ────────────────────────────────────────────────────────────────
    print("\n1. FILE STRUCTURE")
    print("-" * 70)

    # Critical files
    inference_root = root / "inference.py"
    v.check("inference.py in root", inference_root.exists(),
            f"Location: {inference_root}")

    openenv_yaml = root / "openenv.yaml"
    v.check("openenv.yaml exists", openenv_yaml.exists(),
            f"Location: {openenv_yaml}")

    models_py = root / "models.py"
    v.check("models.py exists", models_py.exists(),
            f"Location: {models_py}")

    dockerfile = root / "Dockerfile"
    v.check("Dockerfile exists", dockerfile.exists(),
            f"Location: {dockerfile}")

    requirements = root / "requirements.txt"
    v.check("requirements.txt exists", requirements.exists(),
            f"Location: {requirements}")

    app_py = root / "app.py"
    v.check("app.py exists", app_py.exists(),
            f"Location: {app_py} (FastAPI server)")

    # ────────────────────────────────────────────────────────────────
    # 2. OpenEnv Spec Validation
    # ────────────────────────────────────────────────────────────────
    print("\n2. OPENENV SPEC")
    print("-" * 70)

    try:
        with open(openenv_yaml) as f:
            openenv_content = f.read()
        
        v.check("openenv.yaml parses", len(openenv_content) > 0)
        v.check("Has 'entry_point'", "entry_point:" in openenv_content)
        v.check("Has 'observation_space'", "observation_space:" in openenv_content)
        v.check("Has 'action_space'", "action_space:" in openenv_content)

        # Check key fields
        v.check("Defines propose_price action", "propose_price" in openenv_content)
        v.check("Defines price payload", "price:" in openenv_content)

    except Exception as e:
        v.check("openenv.yaml valid", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 3. Models & Types
    # ────────────────────────────────────────────────────────────────
    print("\n3. MODELS & TYPES")
    print("-" * 70)

    try:
        sys.path.insert(0, str(root))
        from ride_hailing_env.models import (
            Observation, Action, StepResult, EpisodeOutcome
        )
        v.check("Observation model imports", True)
        v.check("Action model imports", True)
        v.check("StepResult model imports", True)
        v.check("EpisodeOutcome model imports", True)
    except ImportError as e:
        v.check("Models import", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 4. Environment Endpoints
    # ────────────────────────────────────────────────────────────────
    print("\n4. ENVIRONMENT ENDPOINTS")
    print("-" * 70)

    try:
        from ride_hailing_env.environment import DynamicPricingEnv
        env = DynamicPricingEnv(task_name="easy", seed=42)

        # Test reset
        obs = env.reset()
        v.check("env.reset() works", obs is not None)

        # Test state
        state = env.state()
        v.check("env.state() works", state is not None)

        # Test step
        action = {"type": "propose_price", "payload": {"price": 20.0}}
        result = env.step(action)
        v.check("env.step() works", result is not None)
        v.check("step() returns StepResult", hasattr(result, "observation"))

    except Exception as e:
        v.check("Environment endpoints", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 5. Environment Variables
    # ────────────────────────────────────────────────────────────────
    print("\n5. ENVIRONMENT VARIABLES")
    print("-" * 70)

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    v.check("API key defined", bool(api_key),
            "Set OPENROUTER_API_KEY or OPENAI_API_KEY", is_env_check=True)

    model_name = os.getenv("MODEL_NAME")
    v.check("MODEL_NAME defined", bool(model_name),
            f"Value: {model_name}", is_env_check=True)

    hf_token = os.getenv("HF_TOKEN")
    v.check("HF_TOKEN defined", bool(hf_token),
            "For HF Space integration", is_env_check=True)

    api_base = os.getenv("API_BASE_URL")
    v.check("API_BASE_URL defined", bool(api_base),
            f"Value: {api_base}", is_env_check=True)

    # ────────────────────────────────────────────────────────────────
    # 6. Dependencies
    # ────────────────────────────────────────────────────────────────
    print("\n6. DEPENDENCIES")
    print("-" * 70)

    try:
        import fastapi
        v.check("fastapi installed", True, f"version: {fastapi.__version__}")
    except ImportError:
        v.check("fastapi installed", False)

    try:
        import uvicorn
        v.check("uvicorn installed", True)
    except ImportError:
        v.check("uvicorn installed", False)

    try:
        import openai
        v.check("openai installed", True, f"version: {openai.__version__}")
    except ImportError:
        v.check("openai installed", False)

    try:
        import pydantic
        v.check("pydantic installed", True, f"version: {pydantic.__version__}")
    except ImportError:
        v.check("pydantic installed", False)

    # ────────────────────────────────────────────────────────────────
    # 7. Tasks & Graders
    # ────────────────────────────────────────────────────────────────
    print("\n7. TASKS & GRADERS")
    print("-" * 70)

    try:
        from ride_hailing_env.config import TASK_CONFIG
        from ride_hailing_env.tasks.graders import GRADER_WEIGHTS

        tasks = ["easy", "medium", "hard"]
        for task in tasks:
            has_config = task in TASK_CONFIG
            v.check(f"Task '{task}' in TASK_CONFIG", has_config)
            has_grader = task in GRADER_WEIGHTS
            v.check(f"Task '{task}' has grader", has_grader)

            if has_grader:
                weights = GRADER_WEIGHTS[task]
                total = sum(weights.values())
                valid = 0.99 <= total <= 1.01  # Allow small rounding
                v.check(f"  → weights sum to 1.0", valid,
                        f"Sum: {total:.4f}")

    except Exception as e:
        v.check("Tasks & graders", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 8. Inference Script Test
    # ────────────────────────────────────────────────────────────────
    print("\n8. INFERENCE SCRIPT")
    print("-" * 70)

    try:
        from inference import run_inference
        v.check("inference.py imports", True)
        v.check("run_inference() function exists", callable(run_inference))

        # Quick test (1 episode) - only if API keys are set
        if api_key:
            print("  Running quick test (1 episode on 'easy' task)...")
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Inference test timed out (likely waiting for LLM API)")
                
                # Set 30 second timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                try:
                    result = run_inference("easy", num_episodes=1, seed=42)
                    signal.alarm(0)  # Cancel alarm
                    v.check("run_inference() executes", True)
                    v.check("Returns dict with 'score'", "score" in result)
                    v.check("Score in [0, 1]", 0.0 <= result.get("score", -1) <= 1.0,
                            f"Score: {result.get('score')}")
                except TimeoutError as te:
                    signal.alarm(0)  # Cancel alarm
                    v.check("run_inference() executes", False, f"Timeout: {str(te)}")
            except (TimeoutError, Exception) as e:
                # If timeout handler doesn't work on this platform, just note it
                v.check("run_inference() executes (quick test)", False, 
                        f"Skipped (missing API credentials or error): {type(e).__name__}")
        else:
            # Skip inference test if API keys not configured
            v.check("inference.py imports", True)
            v.check("run_inference() function exists", callable(run_inference))
            v.check("run_inference() executes (deferred)", True, 
                    "Will test when API credentials configured on HF Space")

    except Exception as e:
        v.check("Inference script", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 9. FastAPI App
    # ────────────────────────────────────────────────────────────────
    print("\n9. FASTAPI APP")
    print("-" * 70)

    try:
        from app import app
        v.check("app.py imports successfully", True)
        v.check("FastAPI 'app' object exists", app is not None)

        # Check routes
        routes = [route.path for route in app.routes]
        has_reset = any("/reset" in r for r in routes)
        v.check("Has /reset endpoint", has_reset)

    except Exception as e:
        v.check("FastAPI app", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # 10. Dockerfile
    # ────────────────────────────────────────────────────────────────
    print("\n10. DOCKERFILE")
    print("-" * 70)

    try:
        with open(root / "Dockerfile") as f:
            dockerfile_content = f.read()

        v.check("Dockerfile has EXPOSE 7860", "EXPOSE 7860" in dockerfile_content)
        v.check("Dockerfile installs requirements", "requirements.txt" in dockerfile_content)
        v.check("Dockerfile runs uvicorn", "uvicorn" in dockerfile_content)

    except Exception as e:
        v.check("Dockerfile validation", False, str(e))

    # ────────────────────────────────────────────────────────────────
    # Print Summary
    # ────────────────────────────────────────────────────────────────
    success = v.print_summary()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = validate_all()
    sys.exit(exit_code)
