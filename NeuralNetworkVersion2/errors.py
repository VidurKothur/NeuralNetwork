def printError(input, message):
    dashStr = len(message) * "*"
    print("\033[98m" + '\n' + dashStr + "\033[00m")
    print("\033[91m" + '\n' + message + '\n' + "\033[00m")
    print("\033[98m" + dashStr + '\n' + "\033[00m")
    print("\033[93m" + 'Input Given:\n' + "\033[00m")
    print("\033[93m" + str(input) + "\033[00m")
    print("\033[98m" + '\n' + dashStr + '\n' + "\033[00m")