""" dynamically load settings

author baiyu
"""
import qfa.cifar100_codebase.conf.global_settings as settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(settings)