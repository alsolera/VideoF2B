# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Andrey Vasilik - basil96@users.noreply.github.com
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

'''This contains the custom command for `setuptools`
that builds the main executable for distribution.'''

import platform
import subprocess

import setuptools

import videof2b.version as version


class BuildPyiExe(setuptools.Command):
    '''Custom command that builds the executable via PyInstaller.
    Based on https://github.com/pypa/setuptools/issues/2591
    '''

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def output_message(self, msg, fp):
        '''Output a message to console and to a file.'''
        print(msg)
        fp.write(f'{msg}\n')
        fp.flush()

    def run(self):
        '''Execute the command.'''
        build_log_name = 'build_exe.log'
        with open(build_log_name, 'w', encoding='utf8') as fp_build:
            messages = [
                f'    System platform: {platform.system()}',
                f'Application version: {version.version}',
                'Building main application EXE via PyInstaller...',
                '=' * 60,
            ]
            for msg in messages:
                self.output_message(msg, fp_build)

            result = subprocess.run(['pyinstaller', '--clean', 'VideoF2B.spec'], stdout=fp_build, stderr=fp_build)

            if result.returncode == 0:
                self.output_message('PyInstaller completed successfully.', fp_build)
            else:
                self.output_message(
                    f'PyInstaller exited with code {result.returncode}. '
                    f'Check build log file {build_log_name} for details.'
                )
