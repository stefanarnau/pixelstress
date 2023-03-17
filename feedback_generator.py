#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:27:35 2023

@author: plkn
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# Number of ids to create files for
ids = range(10)

# Iterate participants
for subject_id in ids:
    
    # Define block condition pattern (fixed for now)
    outcomes = ["good", "bad", "good", "bad", "good", "bad", "good", "bad"]
    the_path = ["close", "close", "easy", "easy", "close", "close", "easy", "easy"]
    
    # Iterate blocks
    for block_nr in range(8):
        
        # Get relevant outcome factor
        outcome_factor = {"good":1,"bad":-1}[outcomes[block_nr]]
        
        # Set final value
        end_point = {"easy": np.random.uniform(0.8, 1.3, (1,)),
                     "close":np.random.uniform(0.05, 0.15, (1,))}[the_path[block_nr]]
        
        # Get performance scores
        seq_scores = np.random.uniform(-1, 1, (30,)) * outcome_factor
        
        # get non-scaled feedback scores
        feedbacks_non_scaled = np.linspace(0, end_point, 30) + seq_scores
        
        # get scaled version of feedback scores (accumulated)
        feedbacks = np.linspace(0, end_point, 30) + np.multiply(seq_scores, np.linspace(0.9, 0, 30))
        
        # Set outcome
        feedbacks[-1] = end_point * outcome_factor
        feedbacks_non_scaled[-1] = end_point * outcome_factor
        
        # Adjust last performance score to match the fixed outcome
        seq_scores[-1] = feedbacks_non_scaled[-1] - feedbacks_non_scaled[-2]
        
        # Get average pixel proportions for sequences
        pixel_proportions = np.linspace(0.5, 0.25, 30)
        
        # Sort pixel proportions by performance scores
        sort_idx = seq_scores.argsort()
        
        # Sort difficulties
        pixel_proportions_sorted = np.zeros(pixel_proportions.shape)
        pixel_proportions_sorted[sort_idx] = pixel_proportions
        
        # Get pixel values for sequences
        pixel_values = []
        for x in pixel_proportions_sorted:
            pixel_values.append(np.random.normal(loc=x, scale=0.08, size=(10,)))
        pixel_values = np.stack(pixel_values)




        a = np.stack((feedbacks, pixel_proportions_sorted)).T

    
    plt.plot(feedbacks_non_scaled)
    plt.hlines(0, 0, 32)
    
