
class Logger():
    def __init__(self, name):
        self.__name = name
    
    def log(self, level, message):
        priority = None
        if (level == 1):
            priority = "INFO"
        elif (level == 2):
            priority = "ERROR"
        else:
            print(f"<LOGGER ERROR> No level provided, message: {message}")

        print(f"[{self.__name}] <{priority}> {message}")
