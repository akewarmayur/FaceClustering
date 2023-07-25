import numpy as np
import os
import pandas as pd
import datetime
from deepface import DeepFace
from deepface.commons import functions
from sklearn.cluster import DBSCAN
from imutils import build_montages
import re
import face_recognition
import pickle
import cv2


class ClusterAlgo:
    def __init__(self):
        pass

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]


    def get_cluster(self, encodings_file_path, data1):
        dest_folder_path = "ClusterImage/"
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)
        df = pd.DataFrame(columns=['Character ID', 'ImagePath'])
        print("[INFO] loading encodings...")
        data = pickle.loads(open(encodings_file_path, "rb").read())
        data = np.array(data)
        print(len(data))
        #
        # if len(data) <= 200:
        #     ms = 3
        #     ep = 0.5
        # elif 200 < len(data) <= 300:
        #     ms = 5
        #     ep = 0.45
        # elif 300 < len(data) <= 500:
        #     ms = 8
        #     ep = 0.42
        # else:
        #     ms = 10
        #     ep = 0.39

        ms = 7
        ep = 0.45

        encodings = [d["encoding"] for d in data]
        nn = [ss["imagePath"] for ss in data1]
        # print(nn)
        # cluster the embeddings
        print("[INFO] clustering...")
        # clt = DBSCAN(metric="euclidean", n_jobs=args["jobs"])
        # eps=3, min_samples=2
        # algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
        # leaf_sizeint, default=30
        # clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.39, min_samples=24)
        clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=ep, min_samples=ms)
        clt.fit(encodings)

        # determine the total number of unique faces found in the dataset
        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("[INFO] # unique faces: {}".format(numUniqueFaces))
        # loop over the unique face integers
        for labelID in labelIDs:
            # find all indexes into the `data` array that belong to the
            # current label ID, then randomly sample a maximum of 25 indexes
            # from the set
            print("[INFO] faces for face ID: {}".format(labelID))
            idxs = np.where(clt.labels_ == labelID)[0]
            a = []
            for i in idxs:
                a.append(nn[i])
            a.sort(key=self.natural_keys)
            tit = "Face ID #{}".format(labelID)
            ti = "Not Clustered Faces" if labelID == -1 else tit
            for zz in a:
                tmp = [ti, zz]
                df_length1 = len(df)
                df.loc[df_length1] = tmp

            idxs = np.random.choice(idxs, size=min(100, len(idxs)),
                                    replace=False)

            faces = []
            for i in idxs:
                try:
                    image = cv2.imread(data[i]["imagePath"])
                    (top, right, bottom, left) = data[i]["loc"]
                    face = image[top:bottom, left:right]
                    face = cv2.resize(face, (112, 112))
                    faces.append(face)
                except:
                    pass

            montage = build_montages(faces, (112, 112), (10, 15))[0]

            # show the output montage
            title = "Face ID #{}".format(labelID)
            title = "Unknown Faces" if labelID == -1 else title
            cv2.imwrite(dest_folder_path + str(title) + ".png", montage)
            # cv2.imshow(title, montage)
            # cv2.waitKey(0)

        return df

    def encode_faces(self, df, fPath):
        savePickleHere = fPath
        if not os.path.exists(savePickleHere):
            os.makedirs(savePickleHere)
        data = []
        data1 = []
        pickle_file_path = savePickleHere + "features.pickle"
        for ind, row in df.iterrows():
            try:
                # load the input image and convert it from RGB (OpenCV ordering)
                # to dlib ordering (RGB)
                print("[INFO] processing image {}/{}".format(ind + 1, len(df)))
                # print(imagePath)
                image = cv2.imread(row['FrameFileName'])
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input image
                # boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
                boxes = [(int(row['FA1']), int(row['FA2']), int(row['FA3']), int(row['FA4']))]

                # compute the facial embedding for the face
                encodings = face_recognition.face_encodings(rgb, boxes)
                # Len of encodings : 128 d-vector

                # build a dictionary of the image path, bounding box location,
                # and facial encodings for the current image
                d = [{"imagePath": row['PaddedFacesPath'], "loc": box, "encoding": enc}
                     for (box, enc) in zip(boxes, encodings)]
                data.extend(d)

                d1 = [{"imagePath": row['FrameFileName'], "loc": box, "encoding": enc}
                      for (box, enc) in zip(boxes, encodings)]
                data1.extend(d1)

            except Exception as e:
                print(row)
                print("Exception in Encoding", e)
                pass
        print("[INFO] serializing encodings...")
        f = open(pickle_file_path, "wb")
        f.write(pickle.dumps(data1))
        f.close()

        f1 = open(savePickleHere + "features_1.pickle", "wb")
        f1.write(pickle.dumps(data))
        f1.close()
        return pickle_file_path, data

    def clusterFaces(self, extract_faces_df):
        Fpath = 'Pickle/'
        pickle_file_path, data = self.encode_faces(extract_faces_df, Fpath)
        print(pickle_file_path)
        df = self.get_cluster(pickle_file_path, data)
        return df
