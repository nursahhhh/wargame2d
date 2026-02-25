import json
from pathlib import Path
from typing import Dict, Any, Optional

from .llm_utils import call_openrouter
from ._prompt_formatter_ import PromptFormatter, PromptConfig
from ..team_intel import TeamIntel  # make sure this is the correct import

class CommanderAgent:
    """
    High-level strategic commander for Wargame2D.
    Produces JSON intent for Captain agent to follow.
    """

    def __init__(self, model: str, api_key: str, run_log_dir: Path):
        self.model = model
        self.api_key = api_key
        self.run_log_dir = run_log_dir

    def decide_intent(self, intel: TeamIntel, step: int, escalation_reason:str, observed_shift: Optional[str] = None,
                    urgency: Optional[int] = None, config: Optional[PromptConfig] = None) -> Dict[str, Any]:
        """
        Given the current TeamIntel and step, produce high-level intent
        """
        # Build Commander prompt using the new _prompt_commander helper
        prompt = self._prompt_commander(intel=intel,
    step=step,
    escalation_reason=escalation_reason,
    observed_shift=observed_shift,
    urgency=urgency,
    config=config
)

        # Call the LLM
        llm_output = call_openrouter(
            prompt=prompt, 
            model=self.model,
            api_key=self.api_key,
            step=step,
            agent_type='commander',
            run_log_dir=self.run_log_dir,
        )


        # 🔥 FIX STARTS HERE
        try:
            if not llm_output.get("tool_calls"):
                raise ValueError("No tool_calls in response")

            tool_call = llm_output["tool_calls"][0]

            if tool_call["function"]["name"] != "set_commander_intent":
                raise ValueError("Unexpected tool call")

            args_raw = tool_call["function"]["arguments"]

            # arguments is a JSON string → parse it
            if isinstance(args_raw, str):
                return json.loads(args_raw)

            # sometimes already parsed
            if isinstance(args_raw, dict):
                return args_raw

            raise ValueError("Invalid arguments format")

        except Exception as e:
            print("Commander parse error:", e)

            return {
                "intent": "hold_position",
                "justification": "fallback due to invalid LLM output",
                "suggested_patterns": []
            }

    def _prompt_commander(self,
                intel: TeamIntel, 
                step: int,
                escalation_reason: Optional[str] = None,
                    observed_shift: Optional[str] = None,
                    urgency: Optional[int] = None,
                    config: Optional[PromptConfig] = None
                ) -> str:
        """
        Build a prompt for the Commander LLM.
        Emphasizes dynamic inference of high-level strategic patterns.
        """
        formatter = PromptFormatter()
        prompt_text, _ = formatter.build_prompt(
            intel=intel,
            step=step,
            allowed_actions={},  # Commander does not need low-level actions
            config=config
        )

        escalation_block = f"""
        ====================================================
        STRATEGIC ESCALATION CONTEXT
        ====================================================
        Reason for Commander Activation:
        {escalation_reason}

        Observed Battlefield Shift:
        {observed_shift or "N/A"}

        Urgency Level: {urgency if urgency is not None else "Normal"} (1 = Low, 5 = Critical)

        Interpret this context carefully. If urgency is high,
        you may shift doctrine more aggressively.
        """


        commander_instructions = """
    You are a tactical commander AI responsible for dynamically inferring the **high-level strategic patterns** for your team.
    Do NOT select low-level moves — focus on overarching strategy that can guide the Captain agent.

    Your goal is to interpret the current game state and suggest strategic patterns that:
    - Maximize force effectiveness
    - Reduce risk to critical units (AWACS, SAM)
    - Exploit enemy weaknesses dynamically
    - Adapt to emerging situations and threats

    Consider for each unit type:
    - Position and capabilities (armed, mobile, radar)
    - Nearby allies and enemies
    - Current pressure, threat scores, and grouping
    - Potential synergies and combined maneuvers

    Respond ONLY in valid JSON with this format:
    {
        "intent": "...",                  # Short identifier for overall strategy
        "justification": "...",           # Explain why this strategy is chosen
        "suggested_patterns": ["..."]     # Tactical guidance for units (dynamic suggestions)
    }

    Suggested patterns can include, but are not limited to:
    - "Split aircraft to explore high-value areas while minimizing exposure"
    - "Keep decoys trailing lead aircraft for distraction"
    - "Converge units for coordinated ambush or multi-angle attack"
    - "Hold key positions to deny enemy access or force chokepoints"
    - "Force enemy to split, then exploit 2v1 advantage"
    - "Reposition SAMs to maximize overlapping radar coverage"
    - "Adapt unit distribution based on real-time enemy behavior"
    """

        full_prompt =f"""
{commander_instructions}

{escalation_block}

GAME STATE:
{prompt_text}
"""
        return full_prompt