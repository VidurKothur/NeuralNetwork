def isType(var, type):
    try:
        if isinstance(var, type):
            return True
        else:
            return False
    except Exception:
        return False

def nonnegative(var):
    if isType(var, (int, float, complex)) and var >= 0:
        return True
    return False

def printError(location, input, message):
    dashStr = len(message) * "*"
    print('\n' + dashStr + '\n')
    print("\033[92m" + 'Location:\n' + "\033[00m")
    print("\033[92m" + str(location) + "\033[00m")
    print('\n' + dashStr)
    print("\033[91m" + '\n' + str(message) + '\n' + "\033[00m")
    print(dashStr + '\n')
    print("\033[93m" + 'Input Given:\n' + "\033[00m")
    print("\033[93m" + str(input) + "\033[00m")
    print('\n' + dashStr + '\n')