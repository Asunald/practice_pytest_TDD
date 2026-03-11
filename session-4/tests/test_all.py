# tests/test_all.py

import pytest
import sys
from pathlib import Path

# Add parent directory to path to import the module
sys.path.insert(0, str(Path(__file__).parent.parent))

from practice_pytest_TDD import (
    find_duplicates,
    difference_set,
    square_tuple,
    merge_dicts,
    ToDo,
    flatten_list_once
)

# List Exercise
def test_find_duplicates():
    assert set(find_duplicates([1,2,2,3,4,4,5])) == {2,4}
    assert find_duplicates([1,2,3]) == []
    assert set(find_duplicates([1,1,1,2,2,3])) == {1,2}

# Set Exercise
def test_difference_set():
    assert difference_set({1,2,3},{2,3,4}) == {1}
    assert difference_set({1},{1}) == set()
    assert difference_set({1,2,3},{4,5,6}) == {1,2,3}

# Tuple Exercise
def test_square_tuple():
    assert square_tuple((1,2,3)) == (1,4,9)
    assert square_tuple(()) == ()
    assert square_tuple((0,5,10)) == (0,25,100)

# Dictionary Exercise
def test_merge_dicts():
    d1 = {'a':1,'b':2}
    d2 = {'b':3,'c':4}
    assert merge_dicts(d1,d2) == {'a':1,'b':5,'c':4}
    assert merge_dicts({},{'a':1}) == {'a':1}
    assert merge_dicts({'x':10},{'y':20}) == {'x':10,'y':20}

# OOP Exercise
def test_todo_class():
    todo = ToDo()
    todo.add_task("task1")
    todo.add_task("task2")
    assert todo.list_tasks() == ["task1","task2"]
    todo.remove_task("task1")
    assert todo.list_tasks() == ["task2"]
    todo.remove_task("task2")
    assert todo.list_tasks() == []

# Flatten list exercise
def test_flatten_list_once():
    assert flatten_list_once([[1,2],[3,4],5]) == [1,2,3,4,5]
    assert flatten_list_once([]) == []
    assert flatten_list_once([[1],[2],[3]]) == [1,2,3]
    assert flatten_list_once([1,2,3]) == [1,2,3]
