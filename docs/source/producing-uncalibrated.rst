######################
Producing Basic videos
######################

To produce a Basic video, follow these steps:

#. Record a flight using a video camera and save the video file to your computer.
#. Open VideoF2B software. The window looks like this when the application starts:

    .. image:: images/main-window.png

#. Choose :menuselection:`File --> Load` in the main menu.
#. Click the :menuselection:`Browse for file` button of the :menuselection:`Video source` box:

    .. image:: images/load-flight-dialog.png

#. Choose the desired video on your computer and click the :guilabel:`Open` button.

    .. image:: images/browse-for-video.png

#. Click the :guilabel:`Load` button or just press the :kbd:`Enter` key.  The video will begin processing in
   the main window.
#. The trace behind the aircraft grows up to 15 seconds long. During processing, you can clear the trace at
   any time by pressing the :kbd:`Space` bar.
#. If you wish to stop processing the video for any reason before VideoF2B finishes tracing it, press the
   :kbd:`Esc` key on the keyboard.  This will stop the tracing, and the result will be a partially processed
   video.
#. You will find the traced video file in the same location as the original video. The traced video will have
   the same name as the original, but with a ``_out`` suffix.  For example, if your original video is named
   ``Flight 1.mp4``, the traced video will be named ``Flight 1_out.mp4``.

To produce Augmented-Reality videos, first you will need to :doc:`calibrate your camera <camera-calibration>`.
