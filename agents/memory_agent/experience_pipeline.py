import argparse
import os
import logging
from memory_agent import MemoryAgent
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openrouter import OpenRouterModelSettings
import asyncio



class LLMClient:
    def __init__(self, model_name="x-ai/grok-4.1-fast"):
        # Use '/' instead of ':' for provider/model
        self.model_str =  "openrouter:x-ai/grok-4.1-fast"

        # Ensure API key is in env
        os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_API_KEY")

        # Initialize the Agent
        self.agent = Agent(model=self.model_str)

    def complete(self, prompt: str) -> str:
        import asyncio

        # Directly pass prompt string to agent.run
        result = asyncio.run(self.agent.run(user_prompt=prompt))
        return result.result  # this part rthrows error fix llm calls !!!!



# ------------------------
# Main function
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Memory Agent Offline Pipeline")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run reflection extraction stage (stage 1)"
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Run experience distillation stage (stage 2)"
    )
    args = parser.parse_args()

    if not args.extract and not args.distill:
        parser.print_help()
        return

    # Models from env or defaults
    """ reflection_model = "deepseek-advanced"
     distill_model = "grok-large"
"""
    # Initialize LLM clients
    reflection_llm = LLMClient()
    distill_llm = LLMClient()

    # -----------------------------
    # Stage 1: Reflection extraction
    # -----------------------------
    if args.extract:
        logging.info("=== Stage 1: Extracting reflections ===")
        agent = MemoryAgent(llm_complete=reflection_llm.complete,
                            episodes_dir="memory/raw",
                            reflections_dir="memory/reflections")
        agent.extract_reflections()
        logging.info("✔ Reflection extraction completed")

    # -----------------------------
    # Stage 2: Experience distillation
    # -----------------------------
    if args.distill:
        logging.info("=== Stage 2: Distilling experience ===")
        agent = MemoryAgent(llm_complete=distill_llm.complete,
                            reflections_dir="memory/reflections",
                            distilled_path="memory/distilled/experience_guidance.json")
        guidance = agent.distill_experience()
        logging.info(f"✔ Distillation completed. Generated {len(guidance['experience_guidance'])} rules")
        logging.info(guidance)


if __name__ == "__main__":
    main()