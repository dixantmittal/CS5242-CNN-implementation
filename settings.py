time_analysis = {}


def print_time_analysis(logger_enabled, logger_level=0):
    global time_analysis
    time_analysis['logger_enabled'] = logger_enabled
    time_analysis['logger_level'] = logger_level
