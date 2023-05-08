#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 16:27:35 2023

@author: plkn
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import itertools as it

# Path out
path_out = "/home/plkn/repos/pixelstress/control_files/"

# Number of ids to create files for
ids_pilot = range(1000, 1020)
ids_experiment = range(100)

# Set parameters
n_blocks = 8
n_sequences = 12
n_trials = 8

# Iterate participants
for subject_id in it.chain(ids_pilot, ids_experiment):

    # Collector for all the numbers
    all_the_lines = []

    # Define block condition pattern (fixed for now)
    outcomes = ["good", "good", "bad", "bad", "good", "good", "bad", "bad"]
    the_path = ["easy", "close", "close", "easy", "close", "easy", "easy", "close"]

    # Get correct response for color 1
    if np.mod(subject_id, 2) == 1:
        correct_idx = 0  # Odd number participants
    else:
        correct_idx = 1  # Even number participants

    # Iterate blocks
    for block_nr in range(n_blocks):

        # Get relevant outcome factor
        if the_path[block_nr] == "easy":
            outcome_factor = {"good": 1, "bad": -1}[outcomes[block_nr]]
        elif the_path[block_nr] == "close":
            outcome_factor = {"good": -1, "bad": 1}[outcomes[block_nr]]

        # Set final value
        end_point = {
            "easy": np.random.uniform(0.35, 0.5, (1,)),
            "close": np.random.uniform(0.01, 0.05, (1,)),
        }[the_path[block_nr]] * outcome_factor

        if the_path[block_nr] == "easy":
            outcome_wiggleroom = 1
        else:
            outcome_wiggleroom = 0

        # Line for blockstart
        all_the_lines.append(
            np.array(
                [
                    1,
                    "",
                    "gridBlockStartProc",
                    subject_id + 1,
                    1,
                    block_nr + 1,
                    outcome_factor,
                    outcome_wiggleroom,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        )

        # Calculate scores and feedbacks
        jump_range = 0.5
        last_jump = 100;
        while np.abs(last_jump) > jump_range:
            
            # Get performance scores
            seq_scores = np.random.uniform(end_point - jump_range, end_point + jump_range, (n_sequences, 1))
            
            # Calculate feedbacks
            feedbacks = []
            for score_idx in range(len(seq_scores)):
                feedbacks.append(seq_scores[:score_idx + 1].mean())
    
            # Set outcome
            feedbacks[-1] = end_point
            
            # Set last score to match feedback
            # feedbacks[-1] = (sum(seq_scores[: -1]) + x) / n_sequences
            # feedbacks[-1] * n_sequences = sum(seq_scores[: -1]) + x
            seq_scores[-1] = (feedbacks[-1] * n_sequences) - sum(seq_scores[: -1])
            
            # Update last jump
            last_jump = seq_scores[-1][0]

        # Get average pixel proportions for sequences
        pixel_proportions = np.linspace(0.49, 0.25, n_sequences)

        # Sort pixel proportions by performance scores
        sort_idx = seq_scores.reshape(-1).argsort()

        # Sort difficulties
        pixel_proportions_sorted = np.zeros(pixel_proportions.shape)
        pixel_proportions_sorted[sort_idx] = pixel_proportions

        # Get pixel values for sequences
        pixel_values = []
        for x in pixel_proportions_sorted:
            pixel_values.append(np.random.normal(loc=x, scale=0.08, size=(n_trials,)))
        pixel_values = np.stack(pixel_values)

        # Loop sequences
        for sequence_nr in range(pixel_values.shape[0]):

            # Line for sequence start
            all_the_lines.append(
                np.array(
                    [
                        1,
                        "",
                        "gridSequenceStartProc",
                        subject_id + 1,
                        3,
                        block_nr + 1,
                        outcome_factor,
                        outcome_wiggleroom,
                        sequence_nr + 1,
                        pixel_proportions_sorted[sequence_nr],
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                )
            )

            # Loop trials
            for trial_nr in range(pixel_values.shape[1]):

                # Get color for trial difficulty value
                color_difficulty = np.random.randint(1, 3, 1)[0]

                # Determine correct response based on key mapping and dominant color
                if (pixel_values[sequence_nr, trial_nr] < 0.5) & (
                    color_difficulty == 1
                ):
                    correct_key = [4, 6][correct_idx]  # left
                elif (pixel_values[sequence_nr, trial_nr] < 0.5) & (
                    color_difficulty == 2
                ):
                    correct_key = [6, 4][correct_idx]  # right
                elif (pixel_values[sequence_nr, trial_nr] >= 0.5) & (
                    color_difficulty == 1
                ):
                    correct_key = [6, 4][correct_idx]  # right
                elif (pixel_values[sequence_nr, trial_nr] >= 0.5) & (
                    color_difficulty == 2
                ):
                    correct_key = [4, 6][correct_idx]  # left

                # Line for trials
                all_the_lines.append(
                    np.array(
                        [
                            1,
                            "",
                            "gridTrialProc",
                            subject_id + 1,
                            5,
                            block_nr + 1,
                            outcome_factor,
                            outcome_wiggleroom,
                            sequence_nr + 1,
                            pixel_proportions_sorted[sequence_nr],
                            0,
                            0,
                            trial_nr + 1,
                            pixel_values[sequence_nr, trial_nr],
                            color_difficulty,
                            correct_key,
                        ]
                    )
                )

            # Line sequence end
            all_the_lines.append(
                np.array(
                    [
                        1,
                        "",
                        "gridSequenceEndProc",
                        subject_id + 1,
                        4,
                        block_nr + 1,
                        outcome_factor,
                        outcome_wiggleroom,
                        sequence_nr + 1,
                        pixel_proportions_sorted[sequence_nr],
                        seq_scores[sequence_nr, 0],
                        feedbacks[sequence_nr],
                        0,
                        0,
                        0,
                        0,
                    ]
                )
            )

        # Line block end
        all_the_lines.append(
            np.array(
                [
                    1,
                    "",
                    "gridBlockEndProc",
                    subject_id + 1,
                    2,
                    block_nr + 1,
                    outcome_factor,
                    outcome_wiggleroom,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
        )

        # Plot feedback
        plt.plot(feedbacks)
        plt.hlines(0, 0, n_sequences)

    # Stack lines to array
    all_the_lines = np.stack(all_the_lines)

    # Create data frame
    cols = [
        "Weight",
        "Nested",
        "Procedure",
        "subjectID",
        "event_code",
        "block_nr",
        "block_outcome",
        "block_wiggleroom",
        "sequence_nr",
        "sequence_difficulty",
        "sequence_feedback",
        "sequence_feedback_scaled",
        "trial_nr",
        "trial_difficulty",
        "color",
        "correct_response",
    ]
    df = pd.DataFrame(all_the_lines, columns=cols)

    # Save
    fn = os.path.join(path_out, f"control_file_{str(subject_id+1)}_cntr.csv")
    df.to_csv(fn, sep="\t", index=False)

    # Columns in file
    # 01: subject id
    # 02: event_code (1=blockstart, 2=blockend, 3=sequencestart, 4=sequenceend, 5=trial)
    # 03: block_nr
    # 04: block_outcome (-1 = bad, 1 = good)
    # 05: block_wiggleroom
    # 06: sequence_nr
    # 07: sequence_difficulty
    # 08: sequence_feedback
    # 09: sequence_feedback_scaled (accumulated)
    # 10: trial_nr
    # 11: trial_difficulty
    # 12: color (1 = color 1, 2 = color 2)
    # 13: Correct response (1 = left, 2 = right)
