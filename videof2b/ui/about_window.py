# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Alberto Solera Rico - videof2b.dev@gmail.com
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
The dialog window for the "about" information.
'''

from PySide6 import QtCore, QtWidgets
import videof2b.version as version

class AboutDialog(QtWidgets.QDialog):
    '''About window.'''

    def __init__(self, parent) -> None:
        super().__init__(
            parent,
            QtCore.Qt.WindowSystemMenuHint |
            QtCore.Qt.WindowTitleHint |
            QtCore.Qt.WindowCloseButtonHint
        )
        self.setup_ui()
        self.start_btn.clicked.connect(self.accept)
        self.start_btn.clicked.connect(self.reject)

    def setup_ui(self):
        '''Designs the UI.'''
        self.setWindowTitle('About VideoF2B')
        self.setContentsMargins(20, 20, 20, 20)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setObjectName('main_layout')
        self.content_lbl = QtWidgets.QLabel(self)
        
        self.content_lbl.setText(f"""<html>
        <h1 id="videof2b">VideoF2B</h1>
        <p>Author&#39;s blog is <a href="http://videof2b.blogspot.com/">here</a></p>
        <p>Source code repository is <a href="https://github.com/alsolera/VideoF2B/">here</a></p>
        <p>VideoF2B is an open-source desktop application for tracing F2B Control Line Stunt competition flight figures in video.</p>
        <h2 id="version">Version</h2>
        <p>"{version.version}"<p>
        <h2 id="overview">Overview</h2>
        <p>Use this application to trace the path of a control line aircraft as it performs aerobatic maneuvers in F2B competition
        and compare it to regulation figures.</p>
        <p>Authors: Alberto Solera, Andrey Vasilik</p>
        <h2 id="features">Features</h2>
        <ul>
        <li><p>Detects the movement of the aircraft and draw the trace of its centroid in video.</p>
        </li>
        <li><p>Draws an augmented-reality (AR) hemisphere that represents the flight envelope.</p>
        </li>
        <li><p>Displays template figures on the surface of the AR sphere according to
        Section 4.2.15 - &quot;Description of Manoeuvres&quot; of the
        <a href="https://www.fai.org/sites/default/files/ciam/sc4_vol_f2_controlline_21.pdf">FAI Sporting Code (Jan 2021)</a>.
        Maneuvre diagrams are available in
        <a href="https://www.fai.org/sites/default/files/ciam/sc4_vol_f2_controlline_annex_4j_21.pdf">Annex 4J (Jan 2021)</a>
        The latest versions of these regulations are available at the
        <a href="https://www.fai.org/page/ciam-code">FAI Sporting Code page</a> under <strong>Section 4 (Aeromodelling)</strong>.</p>
        </li>
        <li><p>Allows the user to rotate and translate the AR sphere during video processing.</p>
        </li>
        <li><p>Includes a utility to perform camera calibration. This enables display of the AR sphere in videos.</p>
        </li>
        <li><p>Includes a utility to estimate the best camera placement in the field.</p>
        </li>
        </ul>
        <h2 id="license">License</h2>
        <p>Copyright (C) 2021 Alberto Solera Rico - videof2b.dev@gmail.com<p>
        <p>Copyright (C) 2021 Andrey Vasilik - basil96</p>
        <p>This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by 
        the Free Software Foundation, either version 3 of the License, or (at your option) any later version.</p>
        <p>This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.</p>
        <p>You should have received a copy of the GNU General Public License along with this program. If not, see <a href="https://www.gnu.org/licenses/">here</a.</p>
        </html>
        """)
        self.content_lbl.setTextFormat(QtCore.Qt.RichText)
        self.content_lbl.setWordWrap(True)
        self.content_lbl.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        self.content_lbl.setOpenExternalLinks(True)
        #
        self.start_btn = QtWidgets.QPushButton('Accept', self)
        self.start_btn.setDefault(True)
        self.start_btn.setEnabled(True)
        #
        self.buttons_layout = QtWidgets.QHBoxLayout()
        self.buttons_layout.addWidget(self.start_btn)
        #
        self.scroll_area = QtWidgets.QScrollArea(widgetResizable=True)
        self.scroll_area.setMinimumHeight(400)
        self.scroll_area.setWidget(self.content_lbl)
        self.main_layout.addWidget(self.scroll_area)
        self.main_layout.addSpacing(20)
        self.main_layout.addLayout(self.buttons_layout)
        
