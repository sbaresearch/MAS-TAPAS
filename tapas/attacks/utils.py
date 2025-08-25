def _verbosed_print(message, verb):
    if verb >1:
        print(message)

def verbosed(x, verb=0):
    if isinstance(x, str):
        _verbosed_print(x, verb)
    else:
        pass