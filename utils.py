import re

def validate_mrz(mrz: str) -> bool:
    pattern = r'([A-Z0-9<]{30}\n?){3}'
    return bool(re.match(pattern, mrz))