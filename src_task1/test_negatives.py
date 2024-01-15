import cv2 as cv
import os

negative_dataset_path = "../train_negative"
positive_dataset_path = "../train_positive"
dataset_path = "../train_images"
actors = ["barney", "betty", "fred", "wilma"]

if __name__ == "__main__":
    for actor in actors[:5]:
        negatives = os.path.join(negative_dataset_path, actor + "_annotations.txt")
        positives = os.path.join(positive_dataset_path, actor + "_annotations.txt")

        negative_bounding_boxes = dict()

        with open(negatives) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                xmin, ymin, xmax, ymax = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                image_name = line[0]
                negative_bounding_boxes[image_name] = negative_bounding_boxes.get(image_name, []) + [(xmin, ymin, xmax, ymax)]

        positive_bounding_boxes = dict()
        with open(positives) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                xmin, ymin, xmax, ymax = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                image_name = line[0]
                positive_bounding_boxes[image_name] = positive_bounding_boxes.get(image_name, []) + [(xmin, ymin, xmax, ymax)]

        for index, image_name in enumerate(negative_bounding_boxes.keys()):
            image = cv.imread(os.path.join(dataset_path, actor, image_name))
            for xmin, ymin, xmax, ymax in negative_bounding_boxes[image_name]:
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            for xmin, ymin, xmax, ymax in positive_bounding_boxes[image_name]:
                cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            cv.imshow("image", image)
            cv.waitKey(0)
            cv.destroyAllWindows()

            if index == 10:
                break

        
        

