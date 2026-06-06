from mdp_simulator.utils.enums import Topics
from enum import EnumMeta


class PubSub:
    _subscribers = None
    _type = None

    def __init__(self, topic_enum=Topics):
        self._subscribers = dict()
        if not isinstance(topic_enum, EnumMeta):
            raise Exception("Setting up a PubSub with an inconsistent enum: received {} expected {}"
                            .format(type(topic_enum), EnumMeta))
        self._type = topic_enum
        pass

    # Public
    def subscribe_multiple(self, topics: list, methods: list):
        for topic in topics:
            for method in methods:
                self.subscribe_single(topic, method)

    def subscribe_single(self, topic, method):
        self.__check_topic(topic)
        self.__check_method(method)

        if self.__check_already_subscribed(topic, method) >= 0:
            raise Exception("Subscribing multiple time the same method to one topic")

        if self._subscribers.get(topic) is None:
            self._subscribers.update({topic: [method]})
        else:
            self._subscribers.get(topic).append(method)

    def emit(self, topic, *args):
        self.__check_topic(topic)

        if self._subscribers.get(topic) is not None:
            for method in self._subscribers.get(topic):
                method(*args)

    def unsubscribe_single(self, topic, method_to_remove):
        self.__check_topic(topic)
        self.__check_method(method_to_remove)
        index_to_pop = self.__check_already_subscribed(topic, method_to_remove)

        if index_to_pop >= 0:
            self._subscribers.get(topic).pop(index_to_pop)
        else:
            raise Exception("Trying to unsubscribe an unsubscribed method")

    def unsubscribe_multiple(self, topics: list, methods: list):
        for topic in topics:
            for method in methods:
                self.unsubscribe_single(topic, method)

    def unsubscribe_all(self):
        self._subscribers = dict()

    # Private
    @staticmethod
    def __check_topic(topic):
        if not isinstance(topic, Topics):
            raise Exception("Subscription can be made only for real Topics")

    @staticmethod
    def __check_method(method):
        if not callable(method):
            raise Exception("Only method can be subscribed to Topics")

    def __check_already_subscribed(self, topic, method_to_check):
        if self._subscribers.get(topic) is not None:
            for i, method in enumerate(self._subscribers.get(topic)):
                if method == method_to_check:
                    return i
        return -1
