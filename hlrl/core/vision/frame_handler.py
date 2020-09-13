import threading
import time
import gc

from d3dshot import D3DShot
from typing import Any, Union, List, Tuple, Optional, Callable
from torch import Tensor

class WindowsFrameHandler():
    """
    Handles a d3dshot instance to return fresh frames.
    """
    def __init__(self, d3dshot: D3DShot,
        transforms: List[Callable[[Any], Any]] = []):
        self.d3dshot = d3dshot
        self.transforms = transforms
        self.latest_polled = False

        # Syncing variables
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

    def _is_latest_new(self) -> bool:
        """
        Returns true if the latest frame has not been polled.
        """
        return (
            not self.latest_polled
            and not self.d3dshot.get_latest_frame() is None
        )

    def is_capturing(self) -> bool:
        """
        Returns true if currently capturing.
        """
        return self.d3dshot.is_capturing

    def get_new_frame(self) -> Any:
        """
        Retrieves a fresh captured frame using d3dshot.
        """
        with self.cond:
            self.cond.wait_for(self._is_latest_new)
            frame = self.d3dshot.get_latest_frame()
            self.latest_polled = True

            return frame

    def get_frame_stack(self, frame_indices: Union[List[int], Tuple[int]],
        stack_dimension: Optional[str] = None):
        """
        Retrieves the stack of frames at the indices provided.

        Args:
            frame_indices ([int], (int,)): The indices of the frames to retrieve
            stack_dimension (str): The dimension to stack the frames in.
        """
        frames = self.d3dshot.get_frame_stack(frame_indices, stack_dimension)

        for transform in self.transforms:
            frames = transform(frames)

        return frames

    def _capture(self, target_fps: int = 60,
        region: Optional[Union[List[int], Tuple[int]]] = None) -> None:
        """
        The thread for the d3dshot capture.

        Args:
            target_fps (int): The target fps of the d3dshot capture.
            region ([int], (int,)): The region to capture.
        """
        self.d3dshot._reset_frame_buffer()

        frame_time = 1 / target_fps

        while self.d3dshot.is_capturing:
            cycle_start = time.time()

            frame = self.d3dshot.display.capture(
                self.d3dshot.capture_output.process,
                region=self.d3dshot._validate_region(region)
            )

            with self.cond:
                self.latest_polled = False
                self.cond.notify()

            if frame is not None:
                self.d3dshot.frame_buffer.appendleft(frame)
            else:
                if len(self.d3dshot.frame_buffer):
                    self.d3dshot.frame_buffer.appendleft(
                        self.d3dshot.frame_buffer[0]
                    )

            gc.collect()

            cycle_end = time.time()

            frame_time_left = frame_time - (cycle_end - cycle_start)

            if frame_time_left > 0:
                time.sleep(frame_time_left)

        self.d3dshot._is_capturing = False

    def capture(self, target_fps: int = 60,
        region: Optional[Union[List[int], Tuple[int]]] = None) -> bool:
        """
        Begins the d3dshot capturing thread, with an extra variable to indicate
        if the latest frame has been polled.

        Args:
            target_fps (int): The target fps of the d3dshot capture.
            region ([int], (int,)): The region to capture.
        """
        target_fps = self.d3dshot._validate_target_fps(target_fps)

        if self.d3dshot.is_capturing:
            return False

        self.d3dshot._is_capturing = True

        self.d3dshot._capture_thread = threading.Thread(
            target=self._capture, args=(target_fps, region)
        )
        self.d3dshot._capture_thread.start()

        return True

    def stop(self) -> bool:
        """
        Stops the capturing thread.
        """
        if not self.d3dshot.is_capturing:
            return False

        self.d3dshot._is_capturing = False

        with self.cond:
            self.latest_polled = False
            self.cond.notify_all()

        if self.d3dshot._capture_thread is not None:
            self.d3dshot._capture_thread.join(timeout=1)
            self.d3dshot._capture_thread = None

        return True
