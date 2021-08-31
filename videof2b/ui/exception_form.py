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

'''The exception dialog form.'''

import logging
import os
import platform
import re

from PySide6 import QtCore, QtGui, QtWidgets

# from openlp.core.version import get_library_versions, get_version
# from openlp.core.widgets.dialogs import FileDialog


log = logging.getLogger(__name__)

class Ui_ExceptionDialog:
    '''The GUI widgets of the exception dialog.'''
    
    def setup_ui(self, exception_dialog):
        '''Set up the UI.'''
        exception_dialog.setObjectName('exception_dialog')
        # exception_dialog.setWindowIcon(UiIcons().main_icon)
        self.exception_layout = QtWidgets.QVBoxLayout(exception_dialog)
        self.exception_layout.setObjectName('exception_layout')
        self.message_layout = QtWidgets.QHBoxLayout()
        self.message_layout.setObjectName('messageLayout')
        # Set margin to make the box a bit wider so the traceback is easier to read. (left, top, right, bottom)
        self.message_layout.setContentsMargins(0, 0, 50, 0)
        self.message_layout.addSpacing(12)
        self.bug_label = QtWidgets.QLabel(exception_dialog)
        # self.bug_label.setPixmap(UiIcons().exception.pixmap(40, 40))
        self.bug_label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.bug_label.setObjectName('bug_label')
        self.message_layout.addWidget(self.bug_label)
        self.message_layout.addSpacing(12)
        self.message_label = QtWidgets.QLabel(exception_dialog)
        self.message_label.setWordWrap(True)
        self.message_label.setObjectName('message_label')
        self.message_layout.addWidget(self.message_label)
        self.exception_layout.addLayout(self.message_layout)
        self.description_explanation = QtWidgets.QLabel(exception_dialog)
        self.description_explanation.setObjectName('description_explanation')
        self.exception_layout.addWidget(self.description_explanation)
        self.description_text_edit = QtWidgets.QPlainTextEdit(exception_dialog)
        self.description_text_edit.setObjectName('description_text_edit')
        self.exception_layout.addWidget(self.description_text_edit)
        self.description_word_count = QtWidgets.QLabel(exception_dialog)
        self.description_word_count.setObjectName('description_word_count')
        self.exception_layout.addWidget(self.description_word_count)
        self.exception_text_edit = QtWidgets.QPlainTextEdit(exception_dialog)
        self.exception_text_edit.setReadOnly(True)
        self.exception_text_edit.setObjectName('exception_text_edit')
        self.exception_layout.addWidget(self.exception_text_edit)
        self.send_report_button = create_button(exception_dialog, 'send_report_button',
                                                icon=UiIcons().email,
                                                click=self.on_send_report_button_clicked)
        self.save_report_button = create_button(exception_dialog, 'save_report_button',
                                                icon=UiIcons().save,
                                                click=self.on_save_report_button_clicked)
        self.attach_file_button = create_button(exception_dialog, 'attach_file_button',
                                                icon=UiIcons().open,
                                                click=self.on_attach_file_button_clicked)
        self.button_box = create_button_box(exception_dialog, 'button_box', ['close'],
                                            [self.send_report_button, self.save_report_button, self.attach_file_button])
        self.exception_layout.addWidget(self.button_box)

        self.retranslate_ui(exception_dialog)
        self.description_text_edit.textChanged.connect(self.on_description_updated)

    # def retranslate_ui(self, exception_dialog):
    #     '''Translate the widgets on the fly.'''
    #     # Note that bugs mail is not clickable, but it adds the blue color and underlining and makes the test copyable.
    #     exception_dialog.setWindowTitle(translate('OpenLP.ExceptionDialog', 'Error Occurred'))
    #     # Explanation text, &nbsp; adds a small space before: If possible, write in English.
    #     self.description_explanation.setText(
    #         translate('OpenLP.ExceptionDialog', '<strong>Please describe what you were trying to do.</strong> '
    #                                             '&nbsp;If possible, write in English.'))
    #     exception_part1 = (translate('OpenLP.ExceptionDialog',
    #                                  '<strong>Oops, OpenLP hit a problem and couldn\'t recover!<br><br>'
    #                                  'You can help </strong> the OpenLP developers to <strong>fix this</strong>'
    #                                  ' by<br> sending them a <strong>bug report to {email}</strong>{newlines}'
    #                                  ).format(email='<a href = "mailto:bugs3@openlp.org" > bugs3@openlp.org</a>',
    #                                           newlines='<br><br>'))
    #     self.message_label.setText(
    #         translate('OpenLP.ExceptionDialog', '{first_part}'
    #                   '<strong>No email app? </strong> You can <strong>save</strong> this '
    #                   'information to a <strong>file</strong> and<br>'
    #                   'send it from your <strong>mail on browser</strong> via an <strong>attachment.</strong><br><br>'
    #                   '<strong>Thank you</strong> for being part of making OpenLP better!<br>'
    #                   ).format(first_part=exception_part1))
    #     self.send_report_button.setText(translate('OpenLP.ExceptionDialog', 'Send E-Mail'))
    #     self.save_report_button.setText(translate('OpenLP.ExceptionDialog', 'Save to File'))
    #     self.attach_file_button.setText(translate('OpenLP.ExceptionDialog', 'Attach File'))


class ExceptionDialog(QtWidgets.QDialog, Ui_ExceptionDialog):
    '''The exception dialog.'''
    def __init__(self):
        '''Constructor.'''
        super().__init__(None, QtCore.Qt.WindowSystemMenuHint | QtCore.Qt.WindowTitleHint)
        self.setup_ui(self)
        self.settings_section = 'crashreport'
        self.report_text = '**OpenLP Bug Report**\n' \
            'Version: {version}\n\n' \
            '--- Details of the Exception. ---\n\n{description}\n\n ' \
            '--- Exception Traceback ---\n{traceback}\n' \
            '--- System information ---\n{system}\n' \
            '--- Library Versions ---\n{libs}\n'

    def exec(self):
        '''Show the dialog.'''
        self.description_text_edit.setPlainText('')
        self.on_description_updated()
        self.file_attachment = None
        return QtWidgets.QDialog.exec(self)

    def _create_report(self):
        '''Create an exception report.'''
        openlp_version = get_version()
        description = self.description_text_edit.toPlainText()
        traceback = self.exception_text_edit.toPlainText()
        system = translate('OpenLP.ExceptionForm', 'Platform: {platform}\n').format(platform=platform.platform())
        library_versions = get_library_versions()
        library_versions['PyUNO'] = self._get_pyuno_version()
        libraries = '\n'.join(['{}: {}'.format(library, version) for library, version in library_versions.items()])

        if is_linux():
            if os.environ.get('KDE_FULL_SESSION') == 'true':
                system += 'Desktop: KDE SC\n'
            elif os.environ.get('GNOME_DESKTOP_SESSION_ID'):
                system += 'Desktop: GNOME\n'
            elif os.environ.get('DESKTOP_SESSION') == 'xfce':
                system += 'Desktop: Xfce\n'
        # NOTE: Keys match the expected input for self.report_text.format()
        return {'version': openlp_version, 'description': description, 'traceback': traceback,
                'system': system, 'libs': libraries}

    def on_save_report_button_clicked(self):
        '''Save exception log and system information to a file.'''
        while True:
            file_path, filter_used = FileDialog.getSaveFileName(
                self,
                translate('OpenLP.ExceptionForm', 'Save Crash Report'),
                self.settings.value(self.settings_section + '/last directory'),
                translate('OpenLP.ExceptionForm', 'Text files (*.txt *.log *.text)'))
            if file_path is None:
                break
            self.settings.setValue(self.settings_section + '/last directory', file_path.parent)
            opts = self._create_report()
            report_text = self.report_text.format(version=opts['version'], description=opts['description'],
                                                  traceback=opts['traceback'], libs=opts['libs'], system=opts['system'])
            try:
                with file_path.open('w') as report_file:
                    report_file.write(report_text)
                    break
            except OSError as e:
                log.exception('Failed to write crash report')
                QtWidgets.QMessageBox.warning(
                    self, translate('OpenLP.ExceptionDialog', 'Failed to Save Report'),
                    translate('OpenLP.ExceptionDialog', 'The following error occurred when saving the report.\n\n'
                                                        '{exception}').format(file_name=file_path, exception=e))

    def on_send_report_button_clicked(self):
        '''Compose an email via system's default email client
        using the exception log and system information.'''
        content = self._create_report()
        source = ''
        exception = ''
        for line in content['traceback'].split('\n'):
            if re.search(r'[/\\]openlp[/\\]', line):
                source = re.sub(r'.*[/\\]openlp[/\\](.*)".*', r'\1', line)
            if ':' in line:
                exception = line.split('\n')[-1].split(':')[0]
        subject = 'Bug report: {error} in {source}'.format(error=exception, source=source)
        mail_urlquery = QtCore.QUrlQuery()
        mail_urlquery.addQueryItem('subject', subject)
        mail_urlquery.addQueryItem('body', self.report_text.format(version=content['version'],
                                                                   description=content['description'],
                                                                   traceback=content['traceback'],
                                                                   system=content['system'],
                                                                   libs=content['libs']))
        if self.file_attachment:
            mail_urlquery.addQueryItem('attach', self.file_attachment)
        mail_to_url = QtCore.QUrl('mailto:clpilotamv@gmail.com')
        mail_to_url.setQuery(mail_urlquery)
        QtGui.QDesktopServices.openUrl(mail_to_url)

    def on_description_updated(self):
        '''Update the minimum number of characters needed in the description.'''
        count = int(20 - len(self.description_text_edit.toPlainText()))
        if count < 0:
            self.__button_state(True)
            self.description_word_count.setText(
                translate('OpenLP.ExceptionDialog', '<strong>Thank you for your description!</strong>'))
        elif count == 20:
            self.__button_state(False)
            self.description_word_count.setText(
                translate('OpenLP.ExceptionDialog', '<strong>Tell us what you were doing when this happened.</strong>'))
        else:
            self.__button_state(False)
            self.description_word_count.setText(
                translate('OpenLP.ExceptionDialog', '<strong>Please enter a more detailed description of the situation'
                                                    '</strong>'))

    def on_attach_file_button_clicked(self):
        '''Attach files to the bug report e-mail.'''
        file_path, filter_used = \
            FileDialog.getOpenFileName(self,
                                       translate('ImagePlugin.ExceptionDialog', 'Select Attachment'),
                                       self.settings.value(self.settings_section + '/last directory'),
                                       '{text} (*)'.format(text=UiStrings().AllFiles))
        log.info('New files {file_path}'.format(file_path=file_path))
        if file_path:
            self.file_attachment = str(file_path)

    def __button_state(self, state):
        '''Toggle the button state.'''
        self.save_report_button.setEnabled(state)
        self.send_report_button.setEnabled(state)
