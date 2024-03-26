##################################
Producing Augmented-Reality videos
##################################

Before you can produce AR videos, you must complete the following:

- :doc:`camera-calibration`
- :doc:`field-setup`
- :doc:`camera-placement`

Then, follow these steps:

#. Choose :menuselection:`File --> Load` in the main menu.

#. Click the :menuselection:`Browse for file` button of the :menuselection:`Video source` box:

    .. image:: images/load-flight-dialog.png

#. Choose the desired video on your computer and click the :guilabel:`Open` button.

    .. image:: images/browse-for-video.png

#. Click the :menuselection:`Browse for file` button of the :menuselection:`Calibration file` box.

#. Choose the calibration file that you created during :doc:`camera calibration </camera-calibration>`.

    .. important::

        Be sure to choose the calibration file that corresponds to the camera and lens that were used to
        record the video you selected above.

    .. note::

        If F2B markers are not available, but you still want to create video that corrects for camera
        distortion (using your camera's calibration file), turn on the option :guilabel:`Skip camera
        location`. Note that in this case, entry of AR-related parameters is disabled and Augmented-Reality
        graphics *will not be drawn*.

#. Enter the following AR-related parameters:

    :guilabel:`Flight radius (m)`: the :term:`flight radius` of the recorded flight, in meters.

    :guilabel:`Height markers: distance to center (m)`: the horizontal distance from the center of the flight
    circle to the F2B markers, in meters.

    :guilabel:`Height markers: height above center of circle (m)`: the elevation of F2B markers above the
    pilot's feet at the center of the flight circle, in meters.

    .. important::
        Please use meters for the above three parameters.

#. Click the :guilabel:`Load` button or just press the :kbd:`Enter` key.

#. The video window will display the first frame of your video so that you can select F2B markers.  This
   procedure locates the camera in video relative to the flight circle so that AR geometry can be displayed.

    .. image:: images/locating-initial-frame-small.png

    Follow the prompts on the status bar to select markers.  Be as accurate as possible when selecting each
    marker.  To **select** a marker, point the mouse cursor to it and click the **left** mouse button. To
    **unselect** the last selected marker, click the **right** mouse button anywhere in the video window.
    
    You will be prompted to select the following four items:

    :guilabel:`Circle center`: select a point **on the ground in the center of the circle**.  If you know that
    the pilot is standing exactly in the center at the start of the video, select a point at his or her feet.
    If the pilot is not standing in the center of the pilot circle at the start of the video, select a point
    on the ground where you estimate the center of the pilot circle to be. This can be done by reviewing the
    video separately in a video player. Fast-forwarding the video to a time when the pilot is in the middle of
    a maneuver is the recommended method of estimating the location of the circle center.

    :guilabel:`Front marker`: select the center of a marker on the far side of the flight circle that is
    nearest to the middle of the video frame.  It does not matter which marker you choose to be the front, as
    long as markers adjacent to it are visible in the video frame.

    :guilabel:`Left marker`: select the center of the nearest marker to the **left** of the front marker on
    the far side of the flight circle, i.e., the next marker in the counterclockwise direction.

    :guilabel:`Right marker`: select the center of the nearest marker to the **right** of the front marker on
    the far side of the flight circle, i.e., the next marker in the clockwise direction.

    .. image:: images/locating-in-progress-small.png

    When you select a marker, VideoF2B draws a small green circle around the selected point. Here is an example of all four markers after selection:

    .. image:: images/locating-example-markers.png

#. When you select the final marker, you will see this prompt:

    .. image:: images/locating-complete-small.png

    If you made incorrect selections, click :guilabel:`No`.  The current marker selections will be cleared,
    and you will have a chance to select all of them again.

    If you are satisfied with your selections, click :guilabel:`Yes`.  Processing will begin.

See :doc:`User Controls <user-controls>` to learn how to control AR geometry during the processing.
