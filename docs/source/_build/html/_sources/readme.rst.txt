papAI package
##########################


**papAI package** is a Python package used to provide a user-friendly interface to use papAI's tools in a python interpreter.


Requirements
------------

The **papAI package** requires Python 3.8 or newer.

Installation
------------

Install from Test PyPi

.. code-block:: text

    pip install -i https://test.pypi.org/simple/ test-papai

Examples
--------

.. code-block:: python

    from papai.pipeline import Pipeline


    class Map:
        def __init__(self):
            self._rbt = red_black_tree.RBTree()

        def __setitem__(self, key, value):
            self._rbt.insert(key=key, data=value)
