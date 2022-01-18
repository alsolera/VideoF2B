# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021-2022  Andrey Vasilik - basil96@users.noreply.github.com
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

'''The exception dialog form.'''

import logging
import os
import platform
from datetime import datetime
from zipfile import ZipFile

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QFontDatabase
from videof2b.core.common import (PD, get_app_metadata, get_lib_versions,
                                  is_linux)
from videof2b.core.common.store import StoreProperties
from videof2b.ui.icons import MyIcons
from videof2b.ui.widgets import FileDialog

log = logging.getLogger(__name__)


class ExceptionDialog(QtWidgets.QDialog, StoreProperties):
    '''User-friendly exception dialog.'''

    def __init__(self, exc_msg):
        super().__init__(None, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)
        self.app_name, self.app_version = get_app_metadata()
        self.mail_addr = 'someone@somewhere.com'
        self.setup_ui()
        self.exception_text_edit.setPlainText(exc_msg)
        self.settings_rpt_dir = 'mru/crashreport_dir'
        self.report_text = '**VideoF2B Bug Report**\n' \
            'Version: {version}\n\n' \
            '--- Exception Details ---\n\n{description}\n\n ' \
            '--- Exception Traceback ---\n{traceback}\n' \
            '--- System information ---\n{system}\n' \
            '--- Library Versions ---\n{libs}\n'
        self.file_attachment = None

    def exec(self):
        '''Show the dialog.'''
        self.on_description_updated()
        return QtWidgets.QDialog.exec(self)

    def _create_report(self):
        '''Create an exception report.'''
        description = self.description_text_edit.toPlainText()
        traceback = self.exception_text_edit.toPlainText()
        system = ('Platform: {platform}\n').format(platform=platform.platform())
        library_versions = get_lib_versions()
        libraries = '\n'.join([
            f'{library}: {version}'
            for library, version in library_versions.items()
        ])

        if is_linux():
            if os.environ.get('KDE_FULL_SESSION') == 'true':
                system += 'Desktop: KDE SC\n'
            elif os.environ.get('GNOME_DESKTOP_SESSION_ID'):
                system += 'Desktop: GNOME\n'
            elif os.environ.get('DESKTOP_SESSION') == 'xfce':
                system += 'Desktop: Xfce\n'
        # NOTE: Keys match the expected input for self.report_text.format()
        return {'version': self.app_version, 'description': description, 'traceback': traceback,
                'system': system, 'libs': libraries}

    def on_save_report_button_clicked(self):
        '''Save exception log and system information to a file.'''
        logs_path = PD.user_data_path
        while True:
            file_name = datetime.strftime(datetime.now(), 'VideoF2B_CrashReport_%Y%m%d_%H%M%S.zip')
            file_path, _ = FileDialog.getSaveFileName(
                self, 'Save Crash Report',
                self.settings.value(self.settings_rpt_dir) / file_name,
                'Zip archive (*.zip)')
            if file_path is None:
                break
            self.settings.setValue(self.settings_rpt_dir, file_path.parent)
            opts = self._create_report()
            report_text = self.report_text.format(**opts)
            try:
                with ZipFile(file_path, 'w') as report_file:
                    report_file.writestr('report.txt', report_text)
                    for log_path in logs_path.glob('*.log'):
                        report_file.write(log_path, log_path.name)
                    log.info(f'User saved crash report to {file_path.name}')
                    QtWidgets.QMessageBox.information(
                        self, 'Thank you!',
                        'Please email the report at your earliest convenience.')
                    break
            except OSError as save_exc:
                log.exception('Failed to write crash report', exc_info=save_exc)
                QtWidgets.QMessageBox.warning(
                    self, 'Failed to Save Report',
                    'The following error occurred when saving the report.\n\n'
                    f'{save_exc}'
                )

    def on_description_updated(self):
        '''Update the minimum number of characters needed in the description.'''
        min_chars = 20
        remaining = int(min_chars - len(self.description_text_edit.toPlainText()))
        if remaining < 0:
            self.__button_state(True)
            self.description_word_count.setText(
                '<strong>Thank you for your description!</strong>')
        elif remaining == min_chars:
            self.__button_state(False)
            self.description_word_count.setText(
                '<strong>Tell us what you were doing when this happened.</strong>')
        else:
            self.__button_state(False)
            self.description_word_count.setText(
                '<strong>Please enter a more detailed description of the situation''</strong>')

    def on_attach_file_button_clicked(self):
        '''Attach files to the bug report e-mail.'''
        file_path, _ = FileDialog.getOpenFileName(
            self,
            'Select Attachment',
            self.settings.value(self.settings_rpt_dir),
            'All files (*)'
        )
        log.info(f'New files {file_path}')
        if file_path:
            self.file_attachment = str(file_path)

    def __button_state(self, state):
        '''Toggle the button state.'''
        self.save_report_button.setEnabled(state)

    def setup_ui(self):
        '''Set up the UI.'''
        self.setObjectName('exception_dialog')
        self.setWindowIcon(MyIcons().videof2b_icon)
        self.exception_layout = QtWidgets.QVBoxLayout(self)
        self.exception_layout.setObjectName('exception_layout')
        self.message_layout = QtWidgets.QHBoxLayout()
        self.message_layout.setObjectName('messageLayout')
        # Widen the box to make the traceback easier to read.
        self.message_layout.setContentsMargins(0, 0, 50, 0)
        self.message_layout.addSpacing(12)
        self.bug_label = QtWidgets.QLabel(self)
        self.bug_label.setPixmap(MyIcons().exception.pixmap(40, 40))
        self.bug_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.bug_label.setObjectName('bug_label')
        self.message_layout.addWidget(self.bug_label)
        self.message_layout.addSpacing(12)
        self.message_label = QtWidgets.QLabel(self)
        self.message_label.setWordWrap(True)
        self.message_label.setObjectName('message_label')
        self.message_layout.addWidget(self.message_label)
        self.exception_layout.addLayout(self.message_layout)
        self.description_explanation = QtWidgets.QLabel(self)
        self.description_explanation.setObjectName('description_explanation')
        self.exception_layout.addWidget(self.description_explanation)
        self.description_text_edit = QtWidgets.QPlainTextEdit(self)
        self.description_text_edit.setObjectName('description_text_edit')
        self.exception_layout.addWidget(self.description_text_edit)
        self.description_word_count = QtWidgets.QLabel(self)
        self.description_word_count.setObjectName('description_word_count')
        self.exception_layout.addWidget(self.description_word_count)
        self.exception_text_edit = QtWidgets.QPlainTextEdit(self)
        self.exception_text_edit.setReadOnly(True)
        self.exception_text_edit.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.exception_text_edit.setObjectName('exception_text_edit')
        self.exception_layout.addWidget(self.exception_text_edit)
        #
        self.save_report_button = QtWidgets.QPushButton(self)
        self.save_report_button.setObjectName('save_report_button')
        self.save_report_button.setIcon(MyIcons().save)
        self.save_report_button.clicked.connect(self.on_save_report_button_clicked)
        #
        self.button_box = QtWidgets.QDialogButtonBox(self)
        self.button_box.setObjectName('button_box')
        self.button_box.setStandardButtons(QtWidgets.QDialogButtonBox.Close)
        self.button_box.addButton(self.save_report_button, QtWidgets.QDialogButtonBox.ActionRole)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.exception_layout.addWidget(self.button_box)

        self.setWindowTitle('Error Occurred')
        self.description_explanation.setText(
            '<strong>Please describe what you were trying to do.</strong> '
            '&nbsp;Write in English if possible.')
        email = f'<a href = "mailto:{self.mail_addr}" >{self.mail_addr}</a>'
        newlines = '<br><br>'
        exception_text = (
            f'"Nearly all the best things that came to me in life have been unexpected, unplanned by me."<br>'
            f'<i>--Carl Sandburg</i>{newlines}'
            f'Unfortunately, this is not one of them.{newlines}'
            f'<strong>VideoF2B encountered a problem and could not recover.{newlines}'
            f'You can help </strong> the VideoF2B developers to <strong>fix this</strong> '
            f'by<br> sending them a <strong>bug report to {email}</strong>{newlines}'
        )
        self.message_label.setText(
            f'{exception_text}'
            f'<strong>No email app? No internet?</strong> You can <strong>save</strong> this '
            f'information to a <strong>file</strong> and<br>'
            f'send it later from your <strong>browser email</strong> via an <strong>attachment.</strong>{newlines}'
            f'<strong>Thank you</strong> for helping to improve VideoF2B!<br>'
        )
        self.save_report_button.setText('Save to File')

        self.description_text_edit.textChanged.connect(self.on_description_updated)
