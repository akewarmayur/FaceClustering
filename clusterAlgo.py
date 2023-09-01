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
        df = pd.DataFrame(columns=['Clusters', 'FacesPath'])
        print("loading features")
        data = pickle.loads(open(encodings_file_path, "rb").read())
        data = np.array(data)
        print(len(data))

        ms = 7
        ep = 0.45

        encodings = [d["encoding"] for d in data]
        nn = [ss["imagePath"] for ss in data1]
        print("clustering...")
        clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=ep, min_samples=ms)
        clt.fit(encodings)
        labelIDs = np.unique(clt.labels_)
        numUniqueFaces = len(np.where(labelIDs > -1)[0])
        print("Cluster faces: {}".format(numUniqueFaces))
        for labelID in labelIDs:
            print("faces for face cluster : {}".format(labelID))
            idxs = np.where(clt.labels_ == labelID)[0]
            a = []
            for i in idxs:
                a.append(nn[i])
            a.sort(key=self.natural_keys)
            tit = "Cluster {}".format(labelID)
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
            title = "Cluster{}".format(labelID)
            title = "Unknown Faces" if labelID == -1 else title
            cv2.imwrite(dest_folder_path + str(title) + ".png", montage)
        return df

    def encode_faces(self, df, fPath):
        savePickleHere = fPath
        if not os.path.exists(savePickleHere):
            os.makedirs(savePickleHere)
        data = []
        data1 = []
        pickle_file_path = savePickleHere + "face_features.pickle"
        for ind, row in df.iterrows():
            try:
                print("processing faces {}/{}".format(ind + 1, len(df)))
                image = cv2.imread(row['FramesPath'])
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = [(int(row['X1']), int(row['X2']), int(row['X3']), int(row['X4']))]
                encodings = face_recognition.face_encodings(rgb, boxes)
                d = [{"imagePath": row['PaddedFacesPath'], "loc": box, "encoding": enc}
                     for (box, enc) in zip(boxes, encodings)]
                data.extend(d)

                d1 = [{"imagePath": row['FramesPath'], "loc": box, "encoding": enc}
                      for (box, enc) in zip(boxes, encodings)]
                data1.extend(d1)

            except Exception as e:
                print("Exception in Feature Extraction", e)
        print("serializing encodings")
        f = open(pickle_file_path, "wb")
        f.write(pickle.dumps(data1))
        f.close()

        f1 = open(savePickleHere + "face_features_1.pickle", "wb")
        f1.write(pickle.dumps(data))
        f1.close()
        return pickle_file_path, data

    def clusterFaces(self, extract_faces_df):
        Fpath = 'PickleFeatures/'
        pickle_file_path, data = self.encode_faces(extract_faces_df, Fpath)
        df = self.get_cluster(pickle_file_path, data)
        return df
