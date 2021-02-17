def create_loud_function(function):
    def loud_function(*args, **kwargs):
        res = function(*args, **kwargs)
        print(f"{function.__name__}(" +
              f"{', '.join([str(x) for x in args])}" +
              (f"{', '.join([str(x) for x in kwargs])}" if kwargs else "") +
              f") = {str(res)}")
        return res
    return loud_function


def create_louder_function(function):
    def louder_function(*args, **kwargs):
        print(f"\n{function.__name__}(" +
              f"{', '.join([str(x) for x in args])}" +
              (f"{', '.join([str(x) for x in kwargs])}" if kwargs else "") +
              ")")
        res = function(*args, **kwargs)
        print(str(res))
        return res
    return louder_function
