################
Loading a Flight
################

To load a flight into VideoF2B for AR processing, follow these steps:

#. Choose :menuselection:`File --> Load` in the main menu.

#. Click the :menuselection:`Browse for file` button of the :menuselection:`Video source` box:

    .. figure:: images/load-flight-dialog.png

        The "Load a Flight" dialog window. Specify all flight parameters here.

#. Choose the desired video on your computer and click the :guilabel:`Open` button.

    .. figure:: images/browse-for-video.png

        File browsing dialog. This may look different on your computer.

#. Click the :menuselection:`Browse for file` button of the :menuselection:`Calibration file` box.

#. Choose the calibration file that you created during :doc:`camera calibration <camera-calibration>`.

    .. important::

        Be sure to choose the calibration file that corresponds to the camera and lens that were used to
        record the video you selected above.

    .. note::

        If F2B markers are not available, but you still want to create video that corrects for camera
        distortion (using your camera's calibration file), turn on the option :guilabel:`Skip camera
        location`. Note that in this case, entry of AR-related parameters is disabled and Augmented-Reality
        graphics *will not be drawn*.

#. Enter the following AR-related parameters:

    Flight radius (m)
        The :term:`flight radius` of the recorded flight, in meters.

    Height markers: distance to center (m)
        The horizontal distance from the center of the flight circle to the F2B markers, in meters.

    Height markers: height above center of circle (m)
        The elevation of F2B markers above the pilot's feet at the center of the flight circle, in meters.

    .. important::
        Please use meters for the above three parameters.

#. Click the :guilabel:`Load` button or just press the :kbd:`Enter` key.
