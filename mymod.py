#!/usr/bin/python
 
"""Show off features of [pydoc] module
 
This is a silly module to
demonstrate docstrings
"""
__author__ =  'David Mertz'
__version__=  '1.0'
__nonsense__ = 'jabberwocky'
 
class MyClass:
    """Demonstrate class docstrings"""
    def __init__ (self, spam=1, eggs=2):
        """Set default attribute values only
 
        Keyword arguments:
        spam -- a processed meat product
        eggs -- a fine breakfast for lumberjacks
        """
        self.spam = spam
        self.eggs = eggs
