import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import os
import time 
import pickle

dataset_path = "../train_positive"
actors = ["barney", "betty", "fred", "wilma"]

def get_aspect_ratios():
    aspect_ratios = dict()

    for actor in actors:
        actor_annotations = os.path.join(dataset_path, actor + "_annotations.txt")
        with open(actor_annotations) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                xmin, ymin, xmax, ymax = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                height = xmax - xmin
                width = ymax - ymin
                who = line[5]
                aspect_ratios[who] = aspect_ratios.get(who, []) + [width / height]
    
    return aspect_ratios

def propose_aspect_ratios():
    actor_aspect_ratios = get_aspect_ratios()
    steps = 5
    aspect_ratios = set()

    for who in actor_aspect_ratios:
        # create a plot for each actor
        plt.figure()
        plt.title(who)
        mean = np.mean(actor_aspect_ratios[who])
        std = np.std(actor_aspect_ratios[who])
        mean = round(mean, 2)
        std = round(std, 2)
        # print(who, mean, std)
        step_size = 2 * std / (steps - 1)
        for i in range(steps):
            value = mean - std + i * step_size
            value = round(value, 2)
            # print(value)
            aspect_ratios.add(value)
        plt.hist(actor_aspect_ratios[who], bins=50)
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(mean + std, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(mean - std, color='k', linestyle='dashed', linewidth=1)
        plt.savefig(who + "_aspect_ratios" + ".png")
        plt.close()

    # print(aspect_ratios)
    return list(aspect_ratios)

def get_widths():
    best_widths = dict()

    for actor in actors:
        actor_annotations = os.path.join(dataset_path, actor + "_annotations.txt")
        with open(actor_annotations) as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                xmin, ymin, xmax, ymax = int(line[1]), int(line[2]), int(line[3]), int(line[4])
                width = ymax - ymin
                who = line[5]
                best_widths[who] = best_widths.get(who, []) + [width]
    
    return best_widths

def propose_widths():
    actor_widths = get_widths()
    steps = 5
    widths = set()

    for who in actor_widths:
        # create a plot for each actor
        plt.figure()
        plt.title(who)
        mean = np.mean(actor_widths[who])
        std = np.std(actor_widths[who])
        mean = round(mean, 2)
        std = round(std, 2)
        # print(who, mean, std)
        step_size = 2 * std / (steps - 1)
        for i in range(steps):
            value = mean - std + i * step_size
            value = round(value, 2)
            # print(value)
            widths.add(value)
        plt.hist(actor_widths[who], bins=50)
        plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(mean + std, color='k', linestyle='dashed', linewidth=1)
        plt.axvline(mean - std, color='k', linestyle='dashed', linewidth=1)
        plt.savefig(who + "_widths" + ".png")
        plt.close()

    # print(widths)
    return list(widths)


def build():
    proposed_aspect_ratios = propose_aspect_ratios()
    proposed_widths = propose_widths()

    with open("proposed_aspect_ratios.pkl", "wb") as f:
        pickle.dump(proposed_aspect_ratios, f)

    with open("proposed_widths.pkl", "wb") as f:
        pickle.dump(proposed_widths, f) 
    

def load():
    with open("proposed_aspect_ratios.pkl", "rb") as f:
        proposed_aspect_ratios = pickle.load(f)

    with open("proposed_widths.pkl", "rb") as f:
        proposed_widths = pickle.load(f) 
    
    return proposed_aspect_ratios, proposed_widths

if __name__ == "__main__":
    # build()
    ar, w = load()
    # print(ar)
    # print(w)
    
    
