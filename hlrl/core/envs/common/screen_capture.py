import numpy as np
import cv2

from mss import mss
from PIL import Image
from threading import Thread, Lock, Condition

class ScreenCapture:
    """
    A module to capture the current screen asynchronously.
    """
    def __init__(self, window_size, res_shape, update_hook = None):
        """
        Creates the screen capture module instance.

        Args:
            window_size (tuple of int): The shape of the window to capture.
            res_shape (tuple of int): The shape to resize the capture to.
            update_hook (function): A function that is called every time a new
                frame is captured.
        """
        self.update_hook = update_hook

        # Window and resizes shape
        self.width, self.height = window_size
        self.res_width, self.res_height, self.res_depth = res_shape
        self.monitor = {
            'top': 0,
            'left': 0,
            'width': self.width,
            'height': self.height
        }

        self.screen = mss()

        # Capture control variables
        self.mut = Lock()
        self.capture_cond = Condition(self.mut)
        self.current_frame = None
        self.screen_polled = False

        self.running = False


    def set_polled(self):
        """
        Sets the current screen to have been polled.
        """
        self.mut.acquire()
        self.screen_polled = True
        self.mut.release()

    def get_screen(self):
        """
        Returns the current frame. Will wait for the first frame if not
        available
        """
        with capture_cond:
            while self.current_frame is None:
                capture_cond.wait()

            return self.current_frame

    def get_next_screen(self):
        """
        Waits and returns the next frame.
        """
        with capture_cond:
            while self.current_frame is None or self.screen_polled:
                capture_cond.wait()

            return self.current_frame

    def start(self):
        """
        Starts capturing the screen.
        """
        self.running = True

        thread = Thread(target = self.update_screen_thread)
        thread.start()
         
    def stop(self):
        """
        Stops capturing the screen.
        """
        self.running = False

        self.mut.acquire()
        self.capture_cond.notifyAll()
        self.mut.release()

    def _get_screen_img(self):
        """
        Captures a frame from the screen.
        """
        im = self.screen.grab(self.monitor)
        im = np.array(im)

        if(self.res_depth == 0 or self.res_depth == 1):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.resize(im, (self.res_width, self.res_height))
            print("Gray", im.shape)
            if(self.res_depth == 1):
                im = np.expand_dims(im, 1)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (self.res_width, self.res_height))
            print("Colour", im.shape)
        return im

    def _update_screen_thread(self):
        """
        The thread to start for capturing. Updates the screen with the new image
        captures. Will call the update hook with the new screen as the parameter
        if a hook is provided.
        """
        while self.running:
            next_frame = self.GetScreenImg()

            self.mut.acquire()
            self.current_frame = next_frame
            self.screen_polled = False
            self.capture_cond.notifyAll()
            self.mut.release()

            self.update_hook(self.current_frame)
