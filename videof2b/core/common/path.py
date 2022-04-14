# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Andrey Vasilik - basil96
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''
Path-related utilities.
'''

from pathlib import Path


def replace_params(args, kwargs, params):
    '''Transform the specified args or kwargs

    :param tuple args: Positional arguments
    :param dict kwargs: Keyword arguments
    :param params: A tuple of tuples with the position
                    and the keyword to replace
    :return: The modified positional and keyword arguments
    :rtype: tuple[tuple, dict]

    Usage:
    Given a method with the following signature,
    assume we want to apply the `str` function to `arg2`:
    `def method(arg1=None, arg2=None, arg3=None)`

    Since `arg2 can be specified positionally as the
    second argument (1 with a zero index) or as a keyword,
    we would call this function as follows:
    `replace_params(args, kwargs, ((1, 'arg2', str),))`
    '''
    args = list(args)
    for position, key_word, transform in params:
        if len(args) > position:
            args[position] = transform(args[position])
        elif key_word in kwargs:
            kwargs[key_word] = transform(kwargs[key_word])
    return tuple(args), kwargs


def path_to_str(path=None):
    '''Convert a Path object or NoneType to a string equivalent.

    :param Path | None path: The value to convert to a string
    :return: An empty string if `path` is None,
             else a string representation of the `path`
    :rtype: str
    '''
    if isinstance(path, str):
        return path
    if not isinstance(path, Path) and path is not None:
        raise TypeError("parameter 'path' must be of type Path or NoneType")
    if path is None:
        return ''
    return str(path)


def str_to_path(string):
    '''Convert a str object to a Path or NoneType.

    This is especially useful because constructing a Path object
    with an empty string causes the Path object to point to the
    current working directory, which is not desirable.

    :param str string: The string to convert
    :return: None if `string` is empty,
             or a Path object representation of `string`
    :rtype: Path | None
    '''
    if not isinstance(string, str):
        return None
    if string == '':
        return None
    return Path(string)


def files_to_paths(file_names):
    '''Convert a list of file names to a list of file Path objects.

    :param list[str] file_names: The list of file names to convert.
    :return: The list converted to file paths
    :rtype: list[Path]
    '''
    if file_names:
        return [str_to_path(file_name) for file_name in file_names]
    return []
