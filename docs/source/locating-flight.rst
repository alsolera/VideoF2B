#################
Locating a Flight
#################


#. The video window will display the first frame of your video so that you can select F2B markers.  This
   procedure locates the camera in video relative to the flight circle so that AR geometry can be displayed.

        .. figure:: images/locating-initial-frame-small.png

            First step of camera locating: begin selecting markers.

   Follow the prompts in the middle of the status bar to select markers.  Be as accurate as possible when
   selecting each marker.

   To **select** a marker, point the mouse cursor to it and click the **left** mouse button.

   To **unselect** the last selected marker, click the **right** mouse button anywhere in the video window.

   You will be prompted to select the following four items:

   Circle center
        Select a point **on the ground in the center of the circle**.  If you know that the pilot is standing
        exactly in the center at the start of the video, select a point at his or her feet.  If the pilot is
        not standing in the center of the pilot circle at the start of the video, select a point on the ground
        where you estimate the center of the pilot circle to be. This can be done by reviewing the video
        separately in a video player. Fast-forwarding the video to a time when the pilot is in the middle of a
        maneuver is the recommended method of estimating the location of the circle center.

   Front marker
        Select the center of a marker on the far side of the flight circle that is nearest to the middle of
        the video frame.  It does not matter which marker you choose to be the front, as long as markers
        adjacent to it are visible in the video frame.

   Left marker
        Select the center of the nearest marker to the **left** of the front marker on the far side of the
        flight circle, i.e., the next marker in the counterclockwise direction.

   Right marker
        Select the center of the nearest marker to the **right** of the front marker on the far side of the
        flight circle, i.e., the next marker in the clockwise direction.

        .. figure:: images/locating-in-progress-small.png

            Camera locating in progress. Center, front, and left markers have been selected in this example.

   When you select a marker, VideoF2B draws a small green circle around the selected point. Here is an example
   of all four markers after selection:

        .. image:: images/locating-example-markers.png

#. When you select the final marker, you will see this prompt:

    .. figure:: images/locating-complete-small.png

        Confirmation prompt at end of camera locating procedure.

    If you made incorrect selections, click :guilabel:`No`.  The current marker selections will be cleared,
    and you will have a chance to select all of them again.

    If you are satisfied with your selections, click :guilabel:`Yes`.  Processing will begin.

See :doc:`user-controls-ar` to learn how to control AR geometry during the processing.
