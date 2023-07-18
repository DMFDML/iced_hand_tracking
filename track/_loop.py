from __future__ import annotations
from curses import flash
from typing import TYPE_CHECKING
from interfaces import Gesture
import numpy as np
import time
import cv2
import mediapipe as mp
import json

if TYPE_CHECKING:
    from .main import HandTracker


def mediapipe_gesture(gesture_index) -> Gesture:
    match gesture_index:
        case "None":
            gesture = Gesture.NONE
        case "Closed_Fist":
            gesture = Gesture.CLOSED_FIST
        case "Open_Palm":
            gesture = Gesture.OPEN_PALM
        case "Pointing_Up":
            gesture = Gesture.POINTING_UP
        case "Thumb_Down":
            gesture = Gesture.THUMB_DOWN
        case "Thumb_Up":
            print("thumbs up")
            gesture = Gesture.THUMB_UP
            print(gesture)
        case "Victory":
            gesture = Gesture.VICTORY
        case "ILoveYou":
            gesture = Gesture.I_LOVE_YOU
        case _:
            print("No None")
            gesture = Gesture.NONE
    return gesture


def pinch_gesture(landmarks) -> Gesture:
    gesture = Gesture.NONE
    thumb = np.array((landmarks[4].x, landmarks[4].y))

    fingers = [
        [8, Gesture.INDEX_PINCH],
        [12, Gesture.MIDDLE_PINCH],
        [16, Gesture.RING_PINCH],
        [20, Gesture.PINKY_PINCH],
    ]

    for idx, g in fingers:
        pos = np.array((landmarks[idx].x, landmarks[idx].y))
        dist = np.linalg.norm(pos - thumb)
        if dist < 0.04:
            gesture = g
    return gesture


def loop(self: HandTracker, frame_rate: int):
    last_handled_frame_timestamp: float = 0.0

    camerasAreOpen = True
    while camerasAreOpen:
        # check the cameras are still accessible
        camerasAreOpen = True
        for camera in self.cameras:
            if not camera["feed"] or not camera["feed"].isOpened():
                camerasAreOpen = False
                break
        if not camerasAreOpen:
            continue

        # Update the frames
        for camera in self.cameras:
            if not camera["feed"]:
                exit()
            success, frame = camera["feed"].read()
            if not success:
                continue
            camera["frame"] = frame

        # Check if it is a frame we would like to deal with
        time_elapsed = time.time() - last_handled_frame_timestamp
        if time_elapsed < 1.0 / frame_rate:
            continue

        last_handled_frame_timestamp = time.time()

        for camera in self.cameras:
            camera["frame"] = cv2.cvtColor(camera["frame"], cv2.COLOR_BGR2RGB)
            if camera["axim"]:
                camera["axim"].set_data(camera["frame"])

        # Reset camera information
        for camera in self.cameras:
            camera["right"]["landmarks"] = []
            camera["left"]["landmarks"] = []
            camera["right"]["gesture"] = Gesture.NONE
            camera["left"]["gesture"] = Gesture.NONE

        if self.recognizer:
            # Process frame with mediapipe.
            for camera in self.cameras:
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=camera["frame"]
                )
                detection_result = self.recognizer.recognize(mp_image)
                for i, hand in enumerate(detection_result.handedness):
                    hand = hand[0]
                    landmarks = detection_result.hand_landmarks[i]

                    hand_key = "right"
                    if hand.category_name == "Left":
                        hand_key = "left"
                    camera[hand_key]["landmarks"] = []
                    camera[hand_key]["gesture"] = Gesture.NONE

                    #### GESTURES ####
                    # TODO: need to check if it is the name or the index and update the match statement accordingly.
                    g = detection_result.gestures[i][0].category_name
                    camera[hand_key]["gesture"] = mediapipe_gesture(g)

                    gesture = pinch_gesture(landmarks)
                    if gesture != Gesture.NONE:
                        camera[hand_key]["gesture"] = gesture

                    #### LANDMARKS ####
                    # TODO: check the landmarks are returned in a consistent order.
                    for landmark in landmarks:
                        pos = [
                            landmark.x * camera["width"],
                            landmark.y * camera["height"],
                        ]
                        camera[hand_key]["landmarks"].append(pos)

        # Update the plot
        for camera in self.cameras:
            camera["right"]["landmarks"] = np.asarray(camera["right"]["landmarks"])
            if camera["right"]["plot"]:
                if camera["right"]["landmarks"].size != 0:
                    camera["right"]["plot"].set_data(
                        camera["right"]["landmarks"][:, 0],
                        camera["right"]["landmarks"][:, 1],
                    )
                else:
                    camera["right"]["plot"].set_data(-1, -1)
            camera["left"]["landmarks"] = np.asarray(camera["left"]["landmarks"])
            if camera["left"]["plot"]:
                if camera["left"]["landmarks"].size != 0:
                    camera["left"]["plot"].set_data(
                        camera["left"]["landmarks"][:, 0],
                        camera["left"]["landmarks"][:, 1],
                    )
                else:
                    camera["left"]["plot"].set_data(-1, -1)

        if self.fig:
            self.fig.canvas.flush_events()

        ## Updating the 3d plot(s)

        right_array = np.empty((21, 2))
        right_array[:] = np.nan
        right_camera_reading_flag = False
        right_camera_readings = [False, False, False]
        for camera in self.cameras:
            if camera["right"]["landmarks"].size != 0:
                right_camera_readings[i] = True
                right_camera_reading_flag = True
                for i, row in enumerate(camera["right"]["landmarks"]):
                    right_array[i] = row[0] / 1000
                    right_array[i] = row[1] / 1000

        left_array = np.empty((21, 2))
        left_array[:] = np.nan
        left_camera_reading_flag = False
        left_camera_readings = [False, False, False]
        for i, camera in enumerate(self.cameras):
            if camera["left"]["landmarks"].size != 0:
                left_camera_readings[i] = True
                left_camera_reading_flag = True
                for j, row in enumerate(camera["left"]["landmarks"]):
                    left_array[j] = row[0] / 1000
                    left_array[j] = row[1] / 1000

        # reset the hands fig
        if self.hands_fig3d:
            self.hands_fig3d_ax.cla()
            self.hands_fig3d_ax.set_xlim(self.x_min, self.x_max)
            self.hands_fig3d_ax.set_xlabel("x")
            self.hands_fig3d_ax.set_ylim(self.y_min, self.y_max)
            self.hands_fig3d_ax.set_ylabel("y (is z)")
            self.hands_fig3d_ax.set_zlim(self.z_min, self.z_max)
            self.hands_fig3d_ax.set_zlabel("z (is y)")

        if right_camera_reading_flag:
            if (
                right_camera_readings[0]
                and right_camera_readings[1]
                and right_camera_readings[2]
            ):
                pos = self.ml_all.predict(right_array)
                break
            elif right_camera_readings[0] and right_camera_readings[1]:
                pos = self.ml_12.predict(right_array)
                break
            elif right_camera_readings[1] and right_camera_readings[2]:
                pos = self.ml_23.predict(right_array)
                break
            elif right_camera_readings[0] and right_camera_readings[2]:
                pos = self.ml_13.predict(right_array)
                break

            if self.path_fig3d:
                if self.path_fig3d_scatter_right:
                    self.path_fig3d_scatter_right.remove()
                    self.path_fig3d_scatter_right = None
                self.path_right = np.vstack([self.path_right, pos[8, :]])
                self.path_fig3d_scatter_right = self.path_fig3d_ax.scatter3D(
                    self.path_right[:, 0],
                    self.path_right[:, 1],
                    self.path_right[:, 2],
                    c="r",
                    marker="o",
                )
            if self.hands_fig3d:
                self.hands_fig3d_ax.plot(pos[0:5, 0], pos[0:5, 1], pos[0:5, 2], "-or")
                self.hands_fig3d_ax.plot(pos[5:9, 0], pos[5:9, 1], pos[5:9, 1], "-or")
                self.hands_fig3d_ax.plot(
                    pos[9:13, 0], pos[9:13, 1], pos[9:13, 0], "-or"
                )
                self.hands_fig3d_ax.plot(
                    pos[13:17, 0], pos[13:17, 1], pos[13:17, 2], "-or"
                )
                self.hands_fig3d_ax.plot(
                    pos[17:21, 0], pos[17:21, 1], pos[17:21, 2], "-or"
                )

        if left_camera_reading_flag:
            if (
                left_camera_readings[0]
                and left_camera_readings[1]
                and left_camera_readings[2]
            ):
                pos = self.ml_all.predict(right_array)
                break
            elif left_camera_readings[0] and left_camera_readings[1]:
                pos = self.ml_12.predict(right_array)
                break
            elif left_camera_readings[1] and left_camera_readings[2]:
                pos = self.ml_23.predict(right_array)
                break
            elif left_camera_readings[0] and left_camera_readings[2]:
                pos = self.ml_13.predict(right_array)
                break

            if self.path_fig3d:
                if self.path_fig3d_scatter_right:
                    self.path_fig3d_scatter_right.remove()
                    self.path_fig3d_scatter_right = None
                self.path_right = np.vstack([self.path_right, pos[8, :]])
                self.path_fig3d_scatter_right = self.path_fig3d_ax.scatter3D(
                    self.path_right[:, 0],
                    self.path_right[:, 1],
                    self.path_right[:, 2],
                    c="b",
                    marker="o",
                )
            if self.hands_fig3d:
                self.hands_fig3d_ax.plot(pos[0:5, 0], pos[0:5, 1], pos[0:5, 2], "-ob")
                self.hands_fig3d_ax.plot(pos[5:9, 0], pos[5:9, 1], pos[5:9, 1], "-ob")
                self.hands_fig3d_ax.plot(
                    pos[9:13, 0], pos[9:13, 1], pos[9:13, 0], "-ob"
                )
                self.hands_fig3d_ax.plot(
                    pos[13:17, 0], pos[13:17, 1], pos[13:17, 2], "-ob"
                )
                self.hands_fig3d_ax.plot(
                    pos[17:21, 0], pos[17:21, 1], pos[17:21, 2], "-ob"
                )

        """

        right_array = np.asarray(right_array)
        _, cols = right_array.shape
        if cols == required_cols:
            pos = self.regression.predict(right_array)
            if self.path_fig3d:
                if self.path_fig3d_scatter_right:
                    self.path_fig3d_scatter_right.remove()
                    self.path_fig3d_scatter_right = None
                self.path_right = np.vstack([self.path_right, pos[8, :]])
                self.path_fig3d_scatter_right = self.path_fig3d_ax.scatter3D(
                    self.path_right[:, 0],
                    self.path_right[:, 1],
                    self.path_right[:, 2],
                    c="r",
                    marker="o",
                )
            if self.hands_fig3d:
                self.hands_fig3d_ax.plot(pos[0:5, 0], pos[0:5, 1], pos[0:5, 2], "-ob")
                self.hands_fig3d_ax.plot(pos[5:9, 0], pos[5:9, 1], pos[5:9, 1], "-ob")
                self.hands_fig3d_ax.plot(
                    pos[9:13, 0], pos[9:13, 1], pos[9:13, 0], "-ob"
                )
                self.hands_fig3d_ax.plot(
                    pos[13:17, 0], pos[13:17, 1], pos[13:17, 2], "-ob"
                )
                self.hands_fig3d_ax.plot(
                    pos[17:21, 0], pos[17:21, 1], pos[17:21, 2], "-ob"
                )

        left_array = np.asarray(left_array)
        _, cols = left_array.shape
        if cols == required_cols:
            pos = self.regression.predict(left_array)
            if self.path_fig3d:
                if self.path_fig3d_scatter_left:
                    self.path_fig3d_scatter_left.remove()
                    self.path_fig3d_scatter_left = None
                self.path_left = np.vstack([self.path_left, pos[8, :]])
                self.path_fig3d_scatter_left = self.path_fig3d_ax.scatter3D(
                    self.path_left[:, 0],
                    self.path_left[:, 1],
                    self.path_left[:, 2],
                    c="b",
                    marker="o",
                )
            if self.hands_fig3d:
                self.hands_fig3d_ax.plot(pos[0:5, 0], pos[0:5, 1], pos[0:5, 2], "-or")
                self.hands_fig3d_ax.plot(pos[5:9, 0], pos[5:9, 1], pos[5:9, 1], "-or")
                self.hands_fig3d_ax.plot(
                    pos[9:13, 0], pos[9:13, 1], pos[9:13, 0], "-or"
                )
                self.hands_fig3d_ax.plot(
                    pos[13:17, 0], pos[13:17, 1], pos[13:17, 2], "-or"
                )
                self.hands_fig3d_ax.plot(
                    pos[17:21, 0], pos[17:21, 1], pos[17:21, 2], "-or"
                )

        if self.path_fig3d:
            self.path_fig3d.canvas.flush_events()
        if self.hands_fig3d:
            self.hands_fig3d.canvas.flush_events()

        #### Create JSON to send to Unity ####

        # To now determine the x, y, z hand positions and send them on.
        tmp_socket_data = {
            "right": {"landmarks": [], "gesture": []},
            "left": {"landmarks": [], "gesture": []},
        }

        right_array = []
        left_array = []
        for i in range(0, 21):
            right_array.append([])
            left_array.append([])

        for camera in self.cameras:
            tmp_socket_data["right"]["gesture"].append(camera["right"]["gesture"])
            tmp_socket_data["left"]["gesture"].append(camera["left"]["gesture"])

            if camera["right"]["landmarks"].size != 0:
                for i, row in enumerate(camera["right"]["landmarks"]):
                    right_array[i].append(row[0])
                    right_array[i].append(row[1])

            if camera["left"]["landmarks"].size != 0:
                for i, row in enumerate(camera["left"]["landmarks"]):
                    left_array[i].append(row[0])
                    left_array[i].append(row[1])

        required_cols = len(self.cameras) * 2
        right_array = np.asarray(right_array)
        _, cols = right_array.shape
        if cols == required_cols:
            pos = self.regression.predict(right_array)
            tmp_socket_data["right"]["landmarks"] = pos

        left_array = np.asarray(left_array)
        _, cols = left_array.shape
        if cols == required_cols:
            pos = self.regression.predict(left_array)
            tmp_socket_data["left"]["landmarks"] = pos

        #### Transform data and send to Unity
        socket_data = {
            "Hand": []
        }
        for i, landmark in enumerate(tmp_socket_data["right"]["landmarks"]):
            socket_data["Hand"].append({
                "handedness": "Right",
                "landmark": i,
                "coords": list(landmark),
                "gesture": tmp_socket_data["right"]["gesture"]
            })
        for i, landmark in enumerate(tmp_socket_data["left"]["landmarks"]):
            socket_data["Hand"].append({
                "handedness": "Left",
                "landmark": i,
                "coords": list(landmark),
                "gesture": tmp_socket_data["left"]["gesture"]
            })

        if self.sock_ip and self.sock_port:
            print(socket_data)
            packet = json.dumps(socket_data)
            print(packet)
            self.sock.sendto(packet.encode(), (self.sock_ip, self.sock_port))
        """
