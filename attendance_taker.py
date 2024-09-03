import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime


# Dlib  / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection to the databaseqq
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()

# Create a table for the current date
current_date = datetime.datetime.now().strftime("%Y_%m_%d")  # Replace hyphens with underscores
table_name = "attendance"
create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (name TEXT, time_in TEXT, time_out TEXT, date DATE, UNIQUE(date))"
cursor.execute(create_table_sql)


# Commit changes and close the connection
conn.commit()
conn.close()

import json

# Load data from the JSON file
with open('data.json', 'r') as json_file:
    student_data = json.load(json_file)

# cc
class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        #  Save the features of faces in the database
        self.face_features_known_list = []
        # / Save the name of faces in the database
        self.face_name_known_list = []

        #  List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        #  cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        #  Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        #  Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

    #  "features_all.csv"  / Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
                print(self.face_features_known_list)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1

        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0
    def get_subject_info(self, student_current_sem_no, current_time):
        conn = sqlite3.connect("attendance.db")# Replace "subject_info.db" with the name of your subject database
        cursor = conn.cursor()
        current_time=9
        end_time=10
        cursor.execute("SELECT sub_name, faculty_name, no_of_hours, timing, sub_code FROM project1 WHERE student_current_sem_no = ? AND start_time = ? AND end_time = ?",
                (student_current_sem_no, current_time, end_time))
        # cursor.execute("SELECT * FROM project1")
        data = cursor.fetchall()       # Fetch all rows from the result set
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        print("Extracted Data:")
        for row in data:
            print(row)
        result = [sublist for sublist in data[0]]  #the code might change according to need
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # subject_info = data  # Assuming only one subject matches the criteria
        conn.close()
        return result
    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # / Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # / Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            #  For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    #  cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        #  / Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)
    # insert data in databaseQ
    cooldown_duration = 5000
    last_check_time = {}
    def attendance(self, name, student_reg_no, student_dept_code, student_degree_code, student_current_sem_no,
                   student_batch_no, Session):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()


        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        # Check if the name has an existing entry for the current date
        subject_info = self.get_subject_info(student_current_sem_no, current_time)
        print(f"+++++++++++++++++++++++ {subject_info}")
        sub_name, faculty_name, no_of_hours, timing, sub_code = subject_info
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        existing_entries = cursor.fetchall()

        if existing_entries:
            # If there are existing entries, count the number of check-ins and check-outs
            check_ins = sum(1 for entry in existing_entries if entry[7] is not None)
            check_outs = sum(1 for entry in existing_entries if entry[8] is not None)

            if check_ins > check_outs:
                # If there are more check-ins than check-outs, perform a check-out
                cursor.execute("UPDATE attendance SET out_time = ? WHERE name = ? AND date = ? AND out_time IS NULL",
                               (current_time, name, current_date))
                conn.commit()
                print(f"{name} checked out for {current_date} at {current_time}")
            else:
                # If there are equal or more check-outs than check-ins, perform a new check-in
                cursor.execute("INSERT INTO attendance (name, student_reg_no, student_dept_code, student_degree_code, "
                               "student_current_sem_no, student_batch_no, in_time, out_time, date,sub_name, faculty_name, no_of_hours, timing, sub_code, Session) VALUES (?, ?, ? , ? ,?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
                           (name, student_reg_no, student_dept_code, student_degree_code, student_current_sem_no,
                            student_batch_no, current_time, current_date, sub_name, faculty_name, no_of_hours, timing, sub_code, Session))
                conn.commit()
                print(f"{name} marked as present for {current_date} at {current_time}")
        else:
            # If there are no existing entries, perform a new check-in
            cursor.execute("INSERT INTO attendance (name, student_reg_no, student_dept_code, student_degree_code, "
                           "student_current_sem_no, student_batch_no, in_time, out_time, date,sub_name, faculty_name, no_of_hours, timing, sub_code, Session) VALUES (?, ?, ? , ? ,?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?)",
                           (name, student_reg_no, student_dept_code, student_degree_code, student_current_sem_no,
                            student_batch_no, current_time, current_date, sub_name, faculty_name, no_of_hours, timing, sub_code, Session))
            conn.commit()
            print(f"{name} marked as present for {current_date} at {current_time}")

        # Update the last check time
        self.last_check_time[name] = time.time()
        print(self.last_check_time)
        conn.close()

    #  Face detection and recognition wit OT from input video stream
    def find_student_data(self, nam):
    # Check if the student with the given name exists in the data
        for student in student_data:
            if student['name'].upper() == nam.upper():
                return student  # Return the student data
        return None  # If not found, return None


    def process(self, stream):
        # 1.  Get faces known from "features.all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # 2.  Detect faces for frame X
                faces = detector(img_rd, 0)

                # 3.  Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4.  Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5.  update frame centroid list
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1  if cnt not changes
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (
                        self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1:   No face cnt changes in this frame!!!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            img_rd = cv2.rectangle(img_rd,
                                                   tuple([d.left(), d.top()]),
                                                   tuple([d.right(), d.bottom()]),
                                                   (255, 255, 255), 2)

                    #  Multi-faces in current frame, use centroid-tracker to track
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()

                    for i in range(self.current_frame_face_cnt):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i],
                                             self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1,
                                             cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2  If cnt of faces changes, 0->1 or 1->0 or ...
                else:
                    logging.debug("scene 2: / Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1  Face cnt decreases: 1->0, 2->1, ...
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  / No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 / Face cnt increase: 0->1, 0->2, ..., 1->2, ...
                    else:
                        logging.debug("  scene 2.2  Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(
                                face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traversal all the faces in the database
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            self.current_frame_face_centroid_list.append(
                                [int(faces[k].left() + faces[k].right()) / 2,
                                 int(faces[k].top() + faces[k].bottom()) / 2])

                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                #
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(
                                        self.current_frame_face_feature_list[k],
                                        self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    #  person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 / Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(
                                min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < 0.4:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("  Face recognition result: %s",
                                              self.face_name_known_list[similar_person_num])

                                # Insert attendance record
                                nam =self.face_name_known_list[similar_person_num]

                                print(type(self.face_name_known_list[similar_person_num]))
                                print('hi',nam)
                                student_info = self.find_student_data(nam)
                                print('data',student_info)
                                if student_info is not None:
                                    # Student data found, store specific information in variables
                                    student_name = student_info['name']
                                    student_reg_no = student_info['reg_no']
                                    student_dept_code = 'AI & DS' if student_info['dept_code'] == 16 else 'other department'
                                    student_degree_code = "B.Tech" if student_info['degree_code'] ==2 else "other degree"
                                    student_current_sem_no = student_info['current_sem_no']
                                    student_batch_no = student_info['batch_no']
                                    Session =student_info['section']
                                    # Add more variables as needed

                                    # Now you can use these variables for further processing
                                    print(f'name: {student_name} \n reg_no: {student_reg_no} \n dept_code: {student_dept_code} \n degree_code:{student_degree_code} \n current_sem_no:{student_current_sem_no} \n batch_no:student_batch_no ')

                                    # Store additional information in separate variables

                                self.attendance(nam,student_reg_no,student_dept_code,student_degree_code,student_current_sem_no,student_batch_no,Session)
                            else:
                                logging.debug("  Face recognition result: Unknown person")

                        # 7.  / Add note on cv2 window
                        self.draw_note(img_rd)

                # 8.  'q'  / Press 'q' to exit
                if kk == ord('q'):
                    break

                self.update_fps()
                cv2.namedWindow("camera", 1)
                cv2.imshow("camera", img_rd)

                logging.debug("Frame ends\n\n")




    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0)              # Get video stream from camera
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()




def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()
