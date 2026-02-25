import os
import json
import requests
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING,List
from dotenv import load_dotenv
from .commander_agent import CommanderAgent
from.llm_utils import call_openrouter
from env.core.actions import Action
from env.core.types import Team, ActionType
from env.world import WorldState

from ..base_agent import BaseAgent
from ..team_intel import TeamIntel
from ..registry import register_agent
from ._prompt_formatter_ import PromptFormatter, PromptConfig

if TYPE_CHECKING:
    from env.environment import StepInfo




# ============================================================
# AGENT
# ============================================================

@register_agent("llm_hybrid_agent")
class LLMHybridAgent(BaseAgent):

    def __init__(
        self,
        team: Team,
        name: str = "LLMHybridAgent",
        api_key: Optional[str] = None,
        model: str = "x-ai/grok-4.1-fast",
        log_file: str = "llm_hybrid_agent_log.jsonl",
        memory_window: int = 5,
        **kwargs,
    ):
        super().__init__(team, name)

        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing")

        self.model = model
        self.prompt_formatter = PromptFormatter()
        self.prompt_config = PromptConfig()
        self.current_strategy = None

        self.memory_window = memory_window
        self.recent_history: list[str] = []

            # ---- RUN LOG FOLDER ----
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_log_dir = Path("llm_runs") / timestamp
        self.run_log_dir.mkdir(parents=True, exist_ok=True)

        self.step_counter = 0


    # --------------------------------------------------------



    def get_actions(
        self,
        state: Dict[str, Any],
        step_info: Optional["StepInfo"] = None,
        **kwargs,
    ) -> tuple[Dict[int, Action], Dict[str, Any]]:

        world: WorldState = state["world"]
        intel: TeamIntel = TeamIntel.build(world, self.team)

        allowed_actions: Dict[int, list[Action]] = {}
        final_actions: Dict[int, Action] = {}

        # -----------------------------
        # Build allowed actions
        # -----------------------------
        for entity in intel.friendlies:
            if not entity.alive:
                continue

            acts = entity.get_allowed_actions(world)
            if acts:
                allowed_actions[entity.id] = acts

        # -----------------------------
        # INITIAL STRATEGY (if missing)
        # -----------------------------
        if self.current_strategy is None:
            print("[INIT] Calling Commander for initial strategy")

            commander = CommanderAgent(
                model=self.model,
                api_key=self.api_key,
                run_log_dir=self.run_log_dir
            )

            self.current_strategy = commander.decide_intent(
                intel=intel,
                step=self.step_counter,
                escalation_reason="Game start – create initial strategy",

            )

        # -----------------------------
        # Build executor prompt
        # -----------------------------
        prompt_text, prompt_payload = self.prompt_formatter.build_prompt(
            intel=intel,
            allowed_actions=allowed_actions,
            config=self.prompt_config,
            step=self.step_counter
        )

        full_prompt = self._build_context_prompt(
            prompt_text,
            self.current_strategy
        )

        # -----------------------------
        # Call Executor LLM
        # -----------------------------
        try:
            llm_response = call_openrouter(
                prompt=full_prompt,
                model=self.model,
                api_key=self.api_key,
                step=self.step_counter,
                agent_type="executor",
                run_log_dir=self.run_log_dir,
            )
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            llm_response = None

        # -----------------------------
        # Parse Tool Response
        # -----------------------------
        parsed_actions = {}

        if llm_response:
            parsed_actions = self._handle_tool_response(
                response=llm_response,
                allowed_actions=allowed_actions,
                state=state,
                step=self.step_counter,
                team_intel=intel
            )

        if parsed_actions:
            final_actions.update(parsed_actions)
        else:
            full_prompt = self._build_context_prompt(
            prompt_text,
            self.current_strategy
        )
            try:
                llm_response = call_openrouter(
                    prompt=full_prompt,
                    model=self.model,
                    api_key=self.api_key,
                    step=self.step_counter,
                    agent_type="executor",
                    run_log_dir=self.run_log_dir,
                )
            except Exception as e:
                print(f"[LLM ERROR] {e}")
                llm_response = None

            parsed_actions = self._handle_tool_response(
                response=llm_response,
                allowed_actions=allowed_actions,
                state=state,
                step=self.step_counter,
                team_intel=intel
            )
            final_actions.update(parsed_actions)

        # -----------------------------
        # Increment step AFTER everything
        # -----------------------------
        self.step_counter += 1

        metadata = {
            "llm_raw_output": llm_response,
            "parsed_actions": parsed_actions,
            "allowed_actions": allowed_actions,
            "prompt_payload": prompt_payload,
            "current_strategy": self.current_strategy,
        }

        return final_actions, metadata
    
    
    def build_experience_advisory_section(
        self,           
        path: str,
        min_confidence: float = 0.7,
        max_rules: int = 8
    ) -> str:
        """
        Reads distilled experience file and converts it into a
        properly formatted LLM advisory section.

        Args:
            file_path: Path to distilled experience JSON file
            min_confidence: Minimum confidence threshold
            max_rules: Maximum number of rules to include

        Returns:
            Formatted advisory string for prompt injection
        """
        BASE_DIR = Path(__file__).resolve().parents[3]  # adjust if needed
        file_path = BASE_DIR / path

        # Prevent crash if file missing
        if not file_path.exists():
            print(f"[WARNING] Experience file not found: {file_path}")
            return ""

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load experience file: {e}")
            return ""

        rules: List[Dict] = data.get("experience_guidance", [])

        # Filter by confidence
        rules = [r for r in rules if r.get("confidence", 0) >= min_confidence]

        # Sort by confidence (highest first)
        rules.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Limit number of rules
        rules = rules[:max_rules]

        if not rules:
            return ""  # No advisory section if nothing qualifies

        # Build advisory text
        advisory_lines = []
        advisory_lines.append("============================================================")
        advisory_lines.append("EXPERIENCE-BASED ADVISORY HEURISTICS")
        advisory_lines.append("============================================================\n")
        advisory_lines.append(
            "The following patterns were extracted from past reflections."
        )
        advisory_lines.append(
            "They are STRATEGIC ADVISORIES and must NOT override:"
        )
        advisory_lines.append("  - HARD constraints")
        advisory_lines.append("  - FIRM constraints")
        advisory_lines.append("  - Mission priorities\n")
        advisory_lines.append("Use them only when multiple valid actions exist.\n")
        advisory_lines.append("Advisories (sorted by confidence):\n")

        for rule in rules:
            confidence = round(rule.get("confidence", 0), 2)
            advisory_lines.append(f"[Confidence: {confidence}]")
            advisory_lines.append(f"Guideline: {rule.get('rule', '')}")
            advisory_lines.append(f"Rationale: {rule.get('rationale', '')}\n")

        return "\n".join(advisory_lines)

    # --------------------------------------------------------
    # PROMPT CONTEXT
    # --------------------------------------------------------

    def _build_context_prompt(self, current_prompt: str, commander_intent: Optional[Dict[str, Any]] = None) -> str:
        history_text = "\n\n".join(self.recent_history[-self.memory_window:])
        experience_avoidance = self.build_experience_advisory_section(
            "wargame2d/memory/distilled/experience_guidance.json"
        )

        
        if commander_intent:
            formatted_intent = json.dumps(commander_intent, indent=2)
            
            commander_intent_text = (
                "\n" + "=" * 60 + "\n"
                "        CURRENT STRATEGIC DIRECTIVE FROM COMMANDER\n"
                + "=" * 60 + "\n"
                f"{formatted_intent}\n"
            )

        combined = f"""
        You are a tactical AI executor controlling friendly units in a 2D combat grid:
        AWACS, Aircraft, Decoys, and SAM sites.

        All cells within range of: Friendly AWACS radar OR active SAM radar
        - This coverage is SHARED among all friendly units instantly

        ============================================================
        MISSION OBJECTIVES (Ordered by Priority)
        ============================================================
        P1. PROTECT FRIENDLY AWACS — Survival is absolute. Never compromise.
        P2. DESTROY ENEMY AWACS — Terminal win condition.
        P3. AVOID DETECTION — Stay outside enemy radar; deny interception.
        P4. GAIN INFORMATION — Expand radar coverage into currently unobserved areas.
        P5. ACHIEVE NUMERICAL ADVANTAGE — Coordinate SAM + aircraft.
        P6. ENGAGE IN COMBAT — Prefer engagements that improve force advantage.

        Priority P4 (Exploration) activates ONLY when:
        - Enemy AWACS is NOT currently detected
        - AND TEAM-UNSEEN cells exist outside friendly sensor coverage

        Lower priorities MUST NOT override higher priorities.
  
        {commander_intent_text} 
        ============================================================
        STRATEGIC ALIGNMENT & MANDATORY COMMAND EXECUTION
        ============================================================

        You operate under the current Commander strategy shown above.

        ------------------------------------------------------------
        STEP 1 — Evaluate Strategic Triggers
        ------------------------------------------------------------

        Before selecting actions, evaluate the following events:

        - ally_unit_destroyed
        - new_enemy_detected
        - base_under_attack
        - heavy_casualties
        - objective_captured
        - sudden_loss_of_air_superiority
        - unexpected_enemy_maneuver
        - strategic_asset_threatened (e.g., AWACS or SAM at risk)

        If ANY of the above is True:

            → You MUST call the Commander for strategy reassessment.
            → Do NOT skip commander call.

        ------------------------------------------------------------
        STEP 2 — If No Strategic Trigger
        ------------------------------------------------------------

        If no strategic trigger is active:

            → The current strategy remains valid

                    
        {experience_avoidance}
        ============================================================
        GAME STATE (Provided Each Turn)
        ============================================================
        You will receive:
        - Grid dimensions and boundaries
        - All friendly unit positions, types, states (armed/unarmed, radar on/off)
        - Known enemy unit positions (if detected)
        - Friendly radar coverage (AWACS + active SAMs)
        - Estimated/known enemy radar coverage
        - Turn number

        ============================================================
        HARD CONSTRAINTS — Never violate under any circumstance
        ============================================================
        [H1] AWACS must NEVER enter known enemy radar coverage
        [H2] AWACS must NEVER end turn with zero safe escape routes
        [H3] Actions that guarantee AWACS destruction next turn are FORBIDDEN
        [H4] Exploration inside friendly radar coverage is INVALID
        [H5] Re-labeling invalid exploration as DEFEND/SUPPORT is FORBIDDEN

    
        ============================================================
        UNIT CAPABILITIES
        ============================================================
        AWACS:
        - Actions: MOVE, WAIT
        - Radar: Provides wide-area friendly sensor coverage
        - Rules: Apply all HARD constraints strictly

        Aircraft:
        - Actions: MOVE, SHOOT (if armed and target in range), WAIT
        - Role: Escort, exploration, interception, terminal attack
        - When enemy AWACS detected: MUST advance or constrain escape

        Decoy:
        - Actions: MOVE, WAIT
        - Role: Expendable scouting, deception, shot absorption
        - May sacrifice for strategic gain, but not for zero value

        SAM:
        - Actions:Shoot, Toggle, Wait.
        - Role: Static area denial; range advantage over aircraft
        - Default: ON (for area control), OFF only for ambush/cooldown

        ============================================================
        SHARED INTELLIGENCE (Team Sensor Fusion)
       ============================================================

        - All friendly units contribute to a single, real-time radar map.
        - There is NO persistent memory of previously observed cells.
        - Visibility is determined solely by current radar coverage of all friendly units
        - Radar coverage is dynamic, as units may move each turn.
        - A cell is considered "currently observed" if it is within any friendly radar range.
        - Cells outside all current radar ranges are considered unobserved.
        - Exploration means moving radar-capable units to expand coverage into these unobserved areas.
        - Units must not violate higher-priority mission objectives or HARD constraints while exploring.

        ============================================================
        OUTPUT FORMAT
        ============================================================
        Respond with valid JSON matching the provided function schema.
        You MUST call exactly ONE function:

        - select_actions → if strategy remains valid
        - request_strategy_update → if strategy need  replannig

        Never call both.

        BEFORE calling the function, you MUST internally evaluate:
        - Threat exposure
        - Engagement advantage
        - Coordination potential
        - Alignment with strategic directive
        For each selected action:

        - Select AT MOST ONE action per unit
        - Include a reason_tag for EVERY action
        - Include a reasoning object for EVERY action with:

            {{
            "threat_level": 0–10,
            "exposure_risk": 0–10,
            "advantage_gain": 0–10,
            "coordination_value": 0–10,
            "justification": "Short tactical explanation"
            }}

        - Omit units that should WAIT (or explicitly include WAIT)

        Actions that increase exposure_risk above 6 without
        advantage_gain above 6 are invalid.

        Exception:
        If no enemy combat aircraft are currently detected,
        exploration-related moves (EXPLORE_UNSEEN or REPOSITION_FOR_EXPLORATION)
        may allow exposure_risk up to 8.


        Allowed reason_tags:
        - PROTECT_AWACS         (escorting, screening, retreating AWACS)
        - INTERCEPT_THREAT      (moving to block/engage approaching enemy)
        - ATTACK_ENEMY_AWACS    (terminal attack execution)
        - EXPLORE_UNSEEN        (advancing into TEAM-UNSEEN space)
        - REPOSITION_FOR_EXPLORATION (moving toward UNSEEN boundary)
        - AREA_DENIAL           (SAM toggling, zone control)
        - DECOY_SACRIFICE       (intentional expendable action)
        - AWAIT_OPPORTUNITY     (justified WAIT with no better option)
        - FORCED_CONSTRAINT_VIOLATION (when no fully legal action exists

        ====================================================
        RECENT HISTORY
        ====================================================
        {history_text}

        ====================================================
        CURRENT SITUATION
        ====================================================
        {current_prompt}

        Respond ONLY with valid JSON.
        """

        self.recent_history.append(current_prompt)

        return combined


    def _extract_entity_id(self, raw):
        if raw is None:
            return None
        if isinstance(raw, int):
            return raw
        match = re.search(r"\d+", str(raw))
        return int(match.group()) if match else None
    

    def _handle_tool_response(self, response, allowed_actions, state, step,team_intel):
        """
        Dispatch based on which tool the LLM called.
        """

        tool_calls = response.get("tool_calls", [])
        if not tool_calls:
            return {}

        tool = tool_calls[0]["function"]
        name = tool["name"]
        args = tool["arguments"]

        if name == "select_actions":
            return self._parse_llm_output(args, allowed_actions)

        elif name == "request_strategy_update":
            return self._handle_strategy_update(args, state, step,team_intel)

        else:
            print(f"Unknown tool call: {name}")
            return {}


    def _handle_strategy_update(self, llm_args: str, state, step,team_intel):


        try:
            data = json.loads(llm_args)
        except Exception:
            print("Strategy update parsing failed.")
            return {}

        reason = data.get("reason", "")
        observed_shift = data.get("observed_shift", "")
        urgency = data.get("urgency_level", 3)

        print("\n[STRATEGY UPDATE REQUESTED]")
        print(f"Reason: {reason}")
        print(f"Observed Shift: {observed_shift}")
        print(f"Urgency: {urgency}\n")

        commander = CommanderAgent(
            model=self.model,
            api_key=self.api_key,
            run_log_dir=self.run_log_dir
        )

        self.current_strategy = commander.decide_intent(
           intel= team_intel,
            step=step,
            escalation_reason=reason,
            observed_shift=observed_shift,
            urgency=urgency,
        )

        
        return {}
    
    

    def _parse_llm_output(self, llm_args: str, allowed_actions):
        actions: Dict[int, Action] = {}

        try:
            data = json.loads(llm_args)
        except Exception:
            print("EXCEPTİONTHROWN LİNE 553  from agent class ")
            return {}

        for item in data.get("actions", []):
            ent = self._extract_entity_id(item.get("entity_id"))
            act_name = item.get("action")
            dir_name = item.get("dir")

            if ent is None or ent not in allowed_actions or not act_name:
                continue

            act_name = act_name.upper()
            dir_name = dir_name.upper() if dir_name else None

            for act in allowed_actions[ent]:
                if act.type.name.upper() != act_name:
                    continue

                if act.type == ActionType.MOVE:

                    if act.params.get("dir").name == dir_name:
                        act_dir =act.params.get("dir")
                        if not act_dir.name or act_dir.name.upper() !=dir_name:
                            continue
                        actions[ent] = act
                        break
                else:
                    actions[ent] = act
                    break

        return actions