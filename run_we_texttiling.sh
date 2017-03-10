#!/usr/bin/env
RESPATH=./seg_results/we_texttiling/

for context in 1 3
    do

    for smoothing in 1 3
        do

        for cutoff in 0.5
            do
                python dynamic_seg.py $context $smoothing 2 $cutoff 2 >& $RESPATH/pk_context$context_smooth$smoothing_cutoff$cutoff.txt &
            done

        done

    done