import time

import colorama
from termcolor import colored

from mdp_simulator.utils.enums import LogTypes, LogLocations, Topics
from mdp_simulator.utils.context import Context

# Init the colorama library
colorama.init()

logType = LogTypes.DEBUG
logLocation = LogLocations.LOCAL


def set_log_type(new_log_type):
    global logType
    if type(new_log_type) != LogTypes:
        return
    logType = new_log_type


def list_to_string(value) -> str:
    output = ""
    for index, elem in enumerate(value):
        if index == 0:
            output = elem
            continue
        output = f"{output}, {elem}"
    return output


def log_time():
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    return f"{colored(curr_time, 'yellow')}"


def stat(*args):
    if logType < LogTypes.STAT:
        return

    if logLocation == LogLocations.LOCAL or logLocation == LogLocations.BOTH:
        output = f"{log_time()} {colored('[STAT]', 'magenta', attrs=['bold'])} : {list_to_string(args)}"
        print(output)


def log(*args):
    if logType < LogTypes.INFO:
        return

    if logLocation == LogLocations.LOCAL or logLocation == LogLocations.BOTH:
        output = f"{log_time()} {colored('[INFO]', 'green', attrs=['bold'])} : {list_to_string(args)}"
        print(output)


def error(*args):
    if logType < LogTypes.ERROR:
        return

    if logLocation == LogLocations.LOCAL or logLocation == LogLocations.BOTH:
        output = f"{log_time()} {colored('[ERROR]', 'red', attrs=['bold'])} : {list_to_string(args)}"
        print(output)


def debug(*args):
    if logType < LogTypes.DEBUG:
        return

    if logLocation == LogLocations.LOCAL or logLocation == LogLocations.BOTH:
        output = f"{log_time()} {colored('[DEBUG]', 'blue', attrs=['bold'])} : {list_to_string(args)}"
        print(output)


def subscribe():
    Context.subscribe_single(Topics.SET_LOG_TYPE, set_log_type)
    Context.subscribe_single(Topics.ERROR, error)
    Context.subscribe_single(Topics.STAT, stat)
    Context.subscribe_single(Topics.INFO, log)
    Context.subscribe_single(Topics.DEBUG, debug)


def unsubscribe():
    Context.unsubscribe_single(Topics.SET_LOG_TYPE, set_log_type)
    Context.unsubscribe_single(Topics.ERROR, error)
    Context.unsubscribe_single(Topics.STAT, stat)
    Context.unsubscribe_single(Topics.INFO, log)
    Context.unsubscribe_single(Topics.DEBUG, debug)
