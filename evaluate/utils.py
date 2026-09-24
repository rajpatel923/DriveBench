import json
import re


def convert_text_to_json(input_text, required_keys=["Q", "A"]):
    """Converts a text input to a JSON object, and checks if it contains the required keys.
    """
    try:
        json_data = json.loads(input_text)
        if all(key in json_data for key in required_keys):
            return json_data
        else:
            raise
    except json.JSONDecodeError:
        print('Fail to convert text to json', input_text)
        return None
    
    
def preprocess_answer_yes_no(answer):
    """Preprocesses the answer from a Vision-Language Model (VLM) output
    to extract and clean Yes/No answers.
    """
    
    answer = answer.strip().lower()

    yes_no_patterns = {
        "yes": r"\byes\b",
        "no": r"\bno\b"
    }
    
    if re.search(yes_no_patterns["yes"], answer):
        return "Yes"
    elif re.search(yes_no_patterns["no"], answer):
        return "No"
    return None
    

def preprocess_answer(answer: str) -> str:
    """Preprocesses the answer from a Vision-Language Model (VLM) output
    to extract and clean ABCD answers.
    
    Args:
    - answer (str): The raw output from the VLM model.

    Returns:
    - str: The cleaned answer (A, B, C, or D), or "" if none can be parsed.

    Matches on the original case so the English article "a" is never read as
    option A (the upstream parser lowercased first, mapping any answer that
    contained the word "a" to A).
    """
    if not answer:
        return ""
    answer = answer.strip()

    # 1. Leading choice letter: "C. Going ahead", "(B)", "A:", "D"
    m = re.match(r"^[\(\[]?([A-D])(?:[\.\):\]]|\s|$)", answer)
    if m:
        return m.group(1)

    # 2. Explicit phrasing: "the answer is B", "Option: C", "choice (A)".
    #    Keyword is case-insensitive; the letter must be uppercase so
    #    "the answer is a car" doesn't parse as A.
    m = re.search(r"(?i:answer|option|choice)\s*(?i:is|:)?\s*[\(\[]?([A-D])(?![A-Za-z])",
                  answer)
    if m:
        return m.group(1)

    # 3. First standalone uppercase letter anywhere
    m = re.search(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", answer)
    if m:
        return m.group(1)

    return ""