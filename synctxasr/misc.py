
def int_or_none(value):
    if value.lower() == 'none':
        return None
    return int(value)