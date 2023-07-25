import clip
import torch
from PIL import Image, ImageDraw, ImageFont
from itertools import islice
import glob
from retinaface.RetinaFace import detect_faces
from detectFaces import FaceDetection
import re
import pandas as pd
import os
import cv2
from clusterAlgo import ClusterAlgo
import subprocess
import celebrityList
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class FaceOccurance:

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def atoi(self, text):
        return int(text) if text.isdigit() else text

    def natural_keys(self, text):
        return [self.atoi(c) for c in re.split(r'(\d+)', text)]

    def extract_frames(self, input_video, fps):
        dest_folder_path = "FramesSavedHere/"
        if not os.path.exists(dest_folder_path):
            os.makedirs(dest_folder_path)
        query = "ffmpeg -i " + input_video + " -pix_fmt rgb24 -vf fps=" + str(
            fps) + " " + dest_folder_path + "img_%06d.png"
        response = subprocess.Popen(query, shell=True, stdout=subprocess.PIPE).stdout.read()
        _ = str(response).encode('utf-8')
        frames = []
        for file in glob.glob(dest_folder_path + '/*'):
            frames.append(file)
        return frames

    def add_text_to_image(self, image, text, position=(20, 20), font_size=15, font_color=(12, 242, 24)):
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", font_size)
        draw.text(position, text, font=font, fill=font_color)

    # Function to resize images to a specified size
    def resize_image(self, image_path, target_size):
        image = Image.open(image_path)
        image = image.resize(target_size, Image.ANTIALIAS)
        return image

    def create_collage(self, images, texts, canvas_size=(800, 600), image_size=(100, 100)):
        canvas = Image.new("RGB", canvas_size, color=(255, 255, 255))

        x_offset = 0
        for i, (image_path, text) in enumerate(zip(images, texts)):
            image = self.resize_image(image_path, image_size)
            canvas.paste(image, (x_offset, 0))
            self.add_text_to_image(canvas, text, position=(x_offset + 5, 5))
            x_offset += image_size[0] + 5

        return canvas

    def get_clip_model(self):
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        return model, preprocess

    def get_prediction(self, frame_path, list_of_labels, how_many_predictions, model, preprocess) -> list:
        Highest3Predictions = []
        try:
            text = clip.tokenize(list_of_labels).to(self.device)
            image = preprocess(Image.open(frame_path)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                probs = probs.tolist()[0]
            vv = {}
            for i, j in enumerate(probs):
                vv[list_of_labels[i]] = j
            maxx = {k: v for k, v in sorted(vv.items(), key=lambda item: item[1], reverse=True)}
            Highest3Predictions = list(islice(maxx.items(), how_many_predictions))
            print(f"{frame_path} : {Highest3Predictions}")
        except Exception as e:
            print("Exception in CLIP predictions:", e)

        return Highest3Predictions

    def extractFaces(self, list_of_images):
        saveFacesHere = "Faces/"
        savePaddedFaces = "PaddedFaces/"
        if not os.path.exists(saveFacesHere):
            os.makedirs(saveFacesHere)

        if not os.path.exists(savePaddedFaces):
            os.makedirs(savePaddedFaces)

        df = pd.DataFrame(
            columns=["FrameFileName", "FacesPath", "PaddedFacesPath", "FA1", "FA2", "FA3", "FA4"])

        def get_faces(image_path):
            return detect_faces(image_path)

        for image_path in list_of_images:
            try:
                ry = image_path.split("\\")
                ry = ry[len(ry) - 1]
                z = str(ry).split(".")[0]
            except:
                ry = image_path.split("/")
                ry = ry[len(ry) - 1]
                z = str(ry).split(".")[0]
            print(z)
            try:
                resp = get_faces(image_path)
                print(f"**{image_path} : {resp}")
                img = cv2.imread(image_path)
                image_wid = img.shape[1]
                image_hgt = img.shape[0]
                i = 0
                for key, value in resp.items():
                    tmp = []
                    aa = value['facial_area']
                    FA = [aa[1], aa[2], aa[3], aa[0]]
                    distnce_between_rightleft_eye = abs(
                        value['landmarks']['right_eye'][0] - value['landmarks']['left_eye'][0])
                    if distnce_between_rightleft_eye < 25:
                        pass
                    else:
                        x1, y1, x2, y2 = aa[0], aa[1], aa[2], aa[3]
                        x = x1
                        y = y1
                        w = abs(x2 - x1)
                        h = abs(y2 - y1)
                        crop_img = img[y:y + h, x:x + w]
                        wid = crop_img.shape[1]
                        hgt = crop_img.shape[0]
                        if (x + w + 50) <= image_wid:
                            croped_hight = y + h + 50
                        else:
                            croped_hight = y + h
                        if (y + h + 50) <= image_hgt:
                            croped_width = x + w + 50
                        else:
                            croped_width = x + w
                        crop_img_clip = img[y - 30:croped_hight, x - 30:croped_width]
                        if abs(wid - hgt) < 15:
                            pass
                        else:
                            cv2.imwrite(saveFacesHere + str(z) + "_" + str(i) + '.png', crop_img)
                            try:
                                cv2.imwrite(savePaddedFaces + str(z) + "_" + str(i) + '.png', crop_img_clip)
                            except:
                                cv2.imwrite(savePaddedFaces + str(z) + "_" + str(i) + '.png', crop_img)
                            tmp.append(image_path)
                            tmp.append(saveFacesHere + str(z) + "_" + str(i) + '.png')
                            tmp.append(savePaddedFaces + str(z) + "_" + str(i) + '.png')
                            tmp.append(FA[0])
                            tmp.append(FA[1])
                            tmp.append(FA[2])
                            tmp.append(FA[3])
                            i += 1
                    if len(tmp) != 0:
                        df_length1 = len(df)
                        df.loc[df_length1] = tmp
            except Exception as e:
                # res[image_path] = []
                print('Error in cropping face :', e)
                pass

        return df

    def convert(self, seconds):
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return "%d:%02d:%02d" % (hour, minutes, seconds)

    def imagepath2timestamp(self, image_path):
        image_name = int(image_path.split("img_")[1].split(".")[0])
        return self.convert(image_name)

    def cluster_using_clip(self, celebrity_df):
        grouped_df = celebrity_df.groupby(['Celebrity'])
        s = []
        for key, item in grouped_df:
            a = grouped_df.get_group(key)
            s.append(a)
            # print(grouped_df.get_group(key), "\n\n")
        sss = pd.concat(s, ignore_index=True)
        sss["Character ID"] = ""
        sss["TimeStamp"] = ""
        ll = sss["Celebrity"].tolist()
        prev = ll[0]
        a = 0
        for i, j in enumerate(ll):
            if i == 0:
                prev = ll[i]
                sss["Character ID"].iloc[i] = "Face ID #" + str(a)
            else:
                curr = ll[i]
                b = a
                if prev == curr:
                    if prev == "unknown":
                        sss["Character ID"].iloc[i] = "Not Clustered Faces"
                    else:
                        sss["Character ID"].iloc[i] = "Face ID #" + str(a)
                    # prev = curr
                else:
                    b += 1
                    if curr == "unknown":
                        sss["Character ID"].iloc[i] = "Not Clustered Faces"
                    else:
                        sss["Character ID"].iloc[i] = "Face ID #" + str(b)
                    a = b
                    prev = curr
        col = ["Character ID", "Celebrity", "FacePath", "FrameFileName"]
        sss = sss[col]
        sss.columns = ['Character ID', 'Celebrity', 'FacePath', 'FrameFileName']

        result = pd.DataFrame(columns=['Character ID', 'Celebrity', 'FacePath', 'FrameFileName', "TimeStamp"])
        for ind, row in sss.iterrows():
            tmp = [row['Character ID'], row['Celebrity'], row['FacePath'], row['FrameFileName'],
                   self.imagepath2timestamp(row['FrameFileName'])]
            df_length1 = len(result)
            result.loc[df_length1] = tmp
        return result

    def time_to_seconds(self, time_string):
        # Parse the time string to a datetime object (Assuming format is HH:MM:SS)
        dt_object = datetime.strptime(time_string, "%H:%M:%S")

        # Calculate the total seconds since the start of the day
        seconds = dt_object.hour * 3600 + dt_object.minute * 60 + dt_object.second

        return seconds

    def plot_with_labels(self, x_values, y_values):
        fig, ax = plt.subplots()
        colors = plt.cm.viridis(np.linspace(0, 1, len(y_values)))
        ax.scatter(x_values, y_values, c=colors, marker='o', s=100, edgecolors='black', linewidth=1.2, alpha=0.7)
        ax.set_title("Celebrity Occurrences", fontsize=14)
        ax.set_xlabel("Celebrity", fontsize=12)
        ax.set_ylabel("TimeStamp(Seconds)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', labelsize=10)
        plt.tight_layout()
        plt.savefig("CelebrityOccurrences.png")
        plt.show()

    def startProcess(self, images_path):
        objFD = FaceDetection()
        objClu = ClusterAlgo()
        list_of_prompts = []
        for celebrities in celebrityList.celebrity_list:
            list_of_prompts.append("a photo of " + str(celebrities))
        list_of_prompts.append("a photo of unidentified person")
        list_of_prompts.append("a photo of ")
        model, preprocess = self.get_clip_model()
        results = pd.DataFrame(columns=["FrameFileName", "FacePath", "Celebrity"])
        try:
            isFolder = True
            tm = images_path.split(".")
            if len(tm) > 1:
                isFolder = False
            if isFolder:
                list_of_images = []
                for fi in glob.glob(images_path + "/*"):
                    list_of_images.append(fi)
            else:
                list_of_images = self.extract_frames(images_path, 1)
            list_of_images.sort(key=self.natural_keys)
            extract_faces_df = objFD.extractFaces(list_of_images)
            extract_faces_df.to_csv("facesInfo.csv")
            extract_faces_df = pd.read_csv("facesInfo.csv")
            ee = objClu.clusterFaces(extract_faces_df)
            ee.to_csv("clusteringResults.csv")
            print(ee)

            for ind, row in extract_faces_df.iterrows():
                face_path = row['PaddedFacesPath']
                Highest3Predictions = self.get_prediction(face_path, list_of_prompts,
                                                          3, model, preprocess)
                c1 = Highest3Predictions[0][0]
                s1 = round(100 * Highest3Predictions[0][1], 2)
                if s1 > 70:
                    detected_celebrity = c1.split("a photo of ")[1]
                    df_length1 = len(results)
                    results.loc[df_length1] = [row['FrameFileName'], face_path, detected_celebrity]
                else:
                    df_length1 = len(results)
                    results.loc[df_length1] = [row['FrameFileName'], face_path, "unknown"]
            clusterDF = self.cluster_using_clip(results)
            clusterDF.to_csv("CelebrityCluster.csv")

            clusterDF['Seconds'] = ""
            sorted_df = clusterDF.sort_values(by='TimeStamp', ascending=True)
            sorted_df['Seconds'] = clusterDF["TimeStamp"].apply(self.time_to_seconds)
            x_values = sorted_df["Celebrity"].tolist()
            y_values = sorted_df["Seconds"].tolist()
            self.plot_with_labels(x_values, y_values)

        except Exception as e:
            print(e)


obj = FaceOccurance()
obj.startProcess("test_video.mp4")
