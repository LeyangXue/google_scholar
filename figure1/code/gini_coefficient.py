#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 00:33:43 2026

@author: leyangxue
"""

import sqlite3
import pandas as pd
import numpy as np

def gini_coefficient(x):
    
    """Compute Gini coefficient of array of values"""
    
    total  = 0
    for i, xi in enumerate(x[:-1], 1):
        total  += np.sum(np.abs(xi - x[i:]))
        
    return total / (len(x)**2 * np.mean(x))

def get_top_percent_elements(input_list, p1, p2):
    
    # Sort the list in descending order
    sorted_list = sorted(input_list, reverse=True)
    
    # Calculate the number of elements in the top p1%
    top_p1_count = int(len(sorted_list)*p1)
    top_p2_count = int(len(sorted_list)*p2)

    # Return all elements in the top 1%
    top_range_elements = sorted_list[top_p1_count:top_p2_count]

    return np.array(top_range_elements)

def Citation_Gini(loadpath, savepath, years):
    
    # List of citation metrics
    metrics = ['C1', 'C2', 'C3', 'C4', 'C5', 'C10', 'CT']
    df = pd.DataFrame(columns=['Year'] + metrics)

    for year in years:
        print(f'calculate the gini with year {year}..')
        
        #load the datasets
        citation_journal = pd.read_csv(loadpath + f'/citation_year{year}.csv')
        p_value_dict = {'0-0.005':(0, 0.005), '0.005-0.01':(0.005, 0.01),'0-0.01':(0, 0.01), '0.01-0.02':(0.01, 0.02),'0.02-0.03':(0.02, 0.03),'0.03-0.04':(0.03, 0.04),'0.04-0.05':(0.04, 0.05)}
        
        for p_value in p_value_dict:
            
            #obtain the ratio 
            (p1, p2) = p_value_dict[p_value]
            
            # Calculate the Gini coefficient for each metric
            gini_values = []
            for metric in metrics:
                citations = citation_journal[metric]
                citation_value = citations.to_list()
                range_citation = get_top_percent_elements(citation_value, p1, p2)
                gini = gini_coefficient(range_citation)
                gini_values.append(gini)

            # Append the Gini coefficients to the dataframe
            df = df.append({'Year': year,'Top-p':p_value, **dict(zip(metrics, gini_values))}, ignore_index=True)
            
    df.to_csv(savepath+'/gini/citation_gini_mag_topn.csv',  index=False)

if __name__ == "__main__":

    loadpath = '/home/tmp/leyangx/work11/mag/year_citation_journal'
    savepath = '/home/tmp/leyangx/work11/mag/results'
    
    years = np.arange(1800,2020,1)
    Citation_Gini(loadpath, savepath, years)
    