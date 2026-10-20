"""String equivalence rules used by the published MATH evaluation.

Independent implementation of Hendrycks MATH normalizer transformations, also
used by the AgentBalance reference. This deliberately is not symbolic algebra.
"""

import re

from .adaptive.scoring import boxed


def extract_math(answer):
    if "\\boxed" in answer or "\\fbox" in answer:
        return boxed(answer) or "0"
    if "answer is " in answer:
        return answer.rsplit("answer is ", 1)[1].strip().rstrip(".")
    return answer[-1] if answer else "A"


def normalize_math(text):
    text = text.replace("\n", "").replace(r"\!", "").replace("\\\\", "\\")
    for old, new in (("tfrac", "frac"), ("dfrac", "frac"), (r"\left", ""), (r"\right", ""),
                     (r"^{\circ}", ""), (r"^\circ", ""), (r"\$", ""), (r"\%", "")):
        text = text.replace(old, new)
    if r"\text{ " in text:
        text = text.split(r"\text{ ", 1)[0]
    text = text.replace(" .", " 0.").replace("{.", "{0.")
    if text.startswith("."):
        text = "0"+text
    if text.count("=") == 1 and len(text.split("=")[0]) <= 2:
        text = text.split("=", 1)[1]
    text = re.sub(r"\\sqrt([^\{])", r"\\sqrt{\1}", text).replace(" ", "")
    text = re.sub(r"\\frac([^\{])([^\{])", r"\\frac{\1}{\2}", text)
    text = re.sub(r"\\frac([^\{])(\{)", r"\\frac{\1}\2", text)
    if text == "0.5":
        text = r"\frac{1}{2}"
    if re.fullmatch(r"-?\d+/-?\d+", text):
        a, b = text.split("/")
        if str(int(a)) == a and str(int(b)) == b:
            text = f"\\frac{{{a}}}{{{b}}}"
    return text
