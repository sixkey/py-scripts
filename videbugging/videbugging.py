def create_loud_function(function):
    def loud_function(*args, **kwargs):
        res = function(*args, **kwargs)
        print(f"{function.__name__}(" +
              f"{', '.join([str(x) for x in args])}" +
              (f"{', '.join([str(x) for x in kwargs])}" if kwargs else "") +
              f") = {str(res)}")
        return res
    return loud_function
