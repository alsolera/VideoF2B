##################
Camera Calibration
##################

Before you can produce :doc:`Augmented-Reality <producing-calibrated>` videos, you must calibrate your camera
system. Camera calibration accomplishes two things in one step. First, it calculates distortion parameters of
the camera's optical system. This allows the processor in VideoF2B to "undistort" every video frame so that
straight lines in the real world remain straight in video. Undistorted frames are essential to many image
processing tasks. Second, it establishes a relationship between the size of objects in video versus the size
of the same objects in the real world. This is important for drawing Augmented-Reality geometry of the correct
size and shape in the video.

Calibration involves the recording of a special video and consists of three easy steps. To begin, start
VideoF2B and choose :menuselection:`Tools --> Calibrate camera..` in the main menu. You will see the following
window:

    .. figure:: images/camera-calibration-dialog.png

        Camera calibration dialog.

Obtain the Calibration Pattern
------------------------------

You have two choices for the calibration pattern: **display** it on screen or **print** it to paper. The
recommended method is to print. However, if you do not have access to a printer, displaying it on screen is
also acceptable.

.. important::

    To **print** the pattern you will need a PDF reader application, such as Adobe Acrobat, Foxit, or similar.

    If you decide to **print** the pattern, make sure to **mount it flat** to a suitable piece of cardboard or
    poster board for easy handling while maintaining accuracy.

.. note::

    The absolute size of the pattern is not important. Whether you display it or print it, do not worry about
    its true size. It is only important that the **entire pattern is flat and visible**.

Record the Video
----------------

Record the video using your camera system. For best results, record at **1080p** resolution. The video should
be fairly short; about 30-50 seconds is enough. The pattern should be visible in its entirety throughout the
video. Move and tilt the camera so that you record as many perspectives of the pattern as possible. To see an
example video, **click the thumbnail under Step 2** in the calibration window. An alternative method is to
mount the camera on a tripod, then move and tilt the printed pattern in front of the camera.

.. attention::

    Configure your camera with the **same** video settings that you will use in the field to record the
    flights. This means that your choice of **lens**, its **focal length**, and **video resolution** all
    **must be the same** during calibration and during field recordings. If the focal length is adjustable
    (also known as a "zoom lens"), then you must make sure to set the focal length to the same value during
    field recordings as you did during calibration. When using the camera of a mobile device, always orient
    the device in **landscape** mode (horizontally) and make sure you always choose the same **zoom factor**
    and **video resolution** as you did during calibration. If you neglect to follow this rule, you will get
    unexpected results in your :doc:`Augmented-Reality <producing-calibrated>` videos. This rule does not
    apply to the frame rate of the video.

If you chose the **Display** option for the pattern in **Step 1**, press the :kbd:`Esc` key to return to the
calibration window after recording the video.

Process the Video
-----------------

Transfer the video file to your computer. Under **Step 3**, browse to the file. Finally, press the
:guilabel:`Start` button at the bottom of the window. VideoF2B will begin processing the calibration video in
the main window:

    .. figure:: images/cam-cal-in-process.png

        Main window of VideoF2B showing camera calibration in progress.

As stated in the message window, the calibration process takes a while. The video playback will appear in slow
motion, and it will seem to "skip" and "freeze" at times, but do not fret -- all is well. The calibration
process is computationally intensive.  If you do want to stop the calibration at any time for any reason, just
press the :kbd:`Esc` key. Otherwise, grab a cup of coffee, relax, and wait patiently until the progress bar
reaches 100%. When finished, the video will disappear from the main window, and you will see some information
about the results in the message window:

    .. figure:: images/cam-cal-complete.png

        Main window at end of camera calibration. Take note of the messages in the message window.

If the calibration fails, most likely your video is too short and/or it does not show the complete pattern
from a sufficient number of points of view. In that case, record another video while paying attention to those
details.

If the calibration succeeds, VideoF2B will create a file named ``CamCalibration.npz`` and two image files in
the same folder as the calibration video. The ``CamCalibration.npz`` file is the calibration file for your
camera system. **Do not lose it**. You will need it for producing every Augmented-Reality video of the flights
you will record with your camera. You may also share it with others who have the same camera system as you.

.. admonition:: For the technically inclined…

    The two image files show a sample frame from the calibration video. The image ``calibresult_nocrop.png``
    is a full-size frame that is "undistorted", i.e., straight lines of the pattern should appear straight in
    the image. To achieve this, the calibration process transforms the original frame in such a way that empty
    pixels appear around the edges of the undistorted image, giving the edges a "pincushion" look:
    
    .. figure:: images/calibresult_nocrop.png

        Uncropped calibrated frame.

    The strength of the pincushion effect depends mostly on the distortion inherent to the lens, and on the
    focal length. Wide-angle action cameras typically show a stronger effect than longer lenses.

    The other image file is ``calibresult.png``. It is the same image as the "no-crop" image above, with one
    important difference. It is cropped to the **maximum usable area** so that the empty pixels are no longer
    visible:

    .. figure:: images/calibresult.png

        Cropped calibrated frame.

    Note that this always results in a smaller image than the full-size video frame that you see in the
    camera. In the above examples, the "no-crop" image size is the original Full HD, or 1920x1080 pixels. The
    cropped image size is 1910x1050 pixels. So a total of 10 pixels were lost from the sides, and a total of
    30 pixels from the top and bottom of the original frame. It is important to keep this in mind when placing
    the camera in the field. Give yourself some room, especially at the bottom of the frame, to account for
    the lost pixels. VideoF2B will "upsize" calibrated video to the size of the original input video whenever
    possible, but some pixels around the border of the original video will be lost due to calibration.

Congratulations, you are ready to record Control Line Stunt videos! The next
step is :doc:`field setup <field-setup>`.
