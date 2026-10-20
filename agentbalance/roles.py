"""Task-specific candidate rosters, with independently written instructions.

Role names/order follow AgentBalance_ e5aeac4 prompt sets and experiment defaults.
These are prompt agents, not copies of the authors' tool execution implementations.
"""

from dataclasses import replace

MMLU = {
    "knowledgeable_expert": "Use broad subject knowledge to solve the question. Identify relevant concepts and check the options.",
    "critic": "Independently challenge the proposed answers. Identify unsupported assumptions and explain necessary corrections.",
    "mathematician": "Apply mathematical and quantitative reasoning. Check calculations, definitions, and logical implications.",
    "psychologist": "Analyze relevant psychology, behavior, and cognitive concepts. Distinguish evidence from intuitive guesses.",
    "historian": "Check historical context, chronology, and causal claims relevant to the question.",
    "doctor": "Apply medical and biological knowledge to the benchmark question. Check mechanisms and terminology.",
    "lawyer": "Analyze legal, political, and institutional concepts relevant to the benchmark question.",
    "economist": "Apply economic and financial reasoning. Check incentives, assumptions, and quantitative relationships.",
    "programmer": "Apply computing, engineering, and physical reasoning. Check algorithms and technical claims.",
}
MATH = {
    "math_solver": "Solve the mathematical problem independently, with concise supporting calculations.",
    "mathematical_analyst": "Formulate the mathematical relationships symbolically, then substitute and verify the values.",
    "programming_expert": "Develop a precise computational solution. Supply Python when useful; do not claim to have executed it.",
    "inspector": "Inspect the proposed mathematics for arithmetic mistakes, invalid steps, and overlooked cases.",
}
CODE = {
    "project_manager": "Extract the function requirements and outline a minimal implementation structure. Keep the design concise.",
    "algorithm_designer": "Choose a correct algorithm. Explain its invariants, complexity, and handling of boundary cases.",
    "programming_expert": MATH["programming_expert"],
    "test_analyst": "Identify useful tests, boundary cases, and possible failures. Do not claim tests ran without execution feedback.",
    "bug_fixer": "Review the available implementation and supply corrected Python that preserves the required interface.",
}
AGENT_SETS = ("mmlu", "mmlu-all", "math", "humaneval", "all")


def templates(name):
    if name == "mmlu":
        workers = dict(list(MMLU.items())[:5])
    elif name == "mmlu-all":
        workers = dict(MMLU)
    elif name == "math":
        # The authors instantiate five MathSolver nodes from a four-role cycle.
        workers = {**MATH, "math_solver_2": MATH["math_solver"]}
    elif name == "humaneval":
        workers = dict(CODE)
    elif name == "all":
        workers = {**MMLU, **MATH, **CODE}
    else:
        raise ValueError(f"Unknown agent set: {name}")
    workers = {k: v + " Treat other agents' contributions as fallible. Follow the task's requested answer format."
               for k, v in workers.items()}
    workers["synthesizer"] = (
        "Resolve disagreements and produce the final answer to the original task. "
        "Follow its output format exactly: a single option letter for multiple choice, "
        "a boxed final value when requested for mathematics, or Python completion code for a coding task. "
        "Do not include commentary outside the requested answer."
    )
    return workers


def with_agent_set(config, name):
    return replace(config, agent_templates=templates(name), min_workers=2)
