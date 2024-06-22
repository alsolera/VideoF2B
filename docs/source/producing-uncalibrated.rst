######################
Producing Basic videos
######################

A Basic video contains a colored **trace** of the path of the aircraft.  No additional geometry is drawn.

Here is an example:

    .. image:: images/uncalibrated-example-snap.png

To produce a Basic video, follow these steps:

#. Record a Stunt flight using a video camera. For best results, record at **1080p** resolution. For
   guidelines on how to position the camera in the field, see :doc:`camera-placement`. Save the video file to
   your computer.

#. Start the VideoF2B application. The main window looks like this when the application starts:

    .. figure:: images/main-window.png

        Main window of VideoF2B.

#. Choose :menuselection:`File --> Load` in the main menu.

#. Click the :menuselection:`Browse for file` button of the :menuselection:`Video source` box:

    .. figure:: images/load-flight-dialog.png

        The "Load a Flight" dialog window. Just choose your video file from here.

#. Choose the desired video on your computer and click the :guilabel:`Open` button.

    .. figure:: images/browse-for-video.png

        File browsing dialog. This may look different on your computer.

#. Click the :guilabel:`Load` button or just press the :kbd:`Enter` key.  The video will begin processing in
   the main window.

#. The trace behind the aircraft grows up to 15 seconds long. During processing, you can clear the trace at
   any time by pressing the :kbd:`Space` bar.

#. If you wish to stop processing the video for any reason before VideoF2B finishes tracing it, press the
   :kbd:`Esc` key on the keyboard.  This will stop the tracing, and the result will be a partially processed
   video.

#. When finished, you will find the traced video file in the same location as the original video. The traced
   video will have the same name as the original, but with a ``_out`` suffix.  For example, if your original
   video is named ``Flight 1.mp4``, the traced video will be named ``Flight 1_out.mp4``.
